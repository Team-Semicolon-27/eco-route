"use client";
import { useEffect, useRef, useState } from 'react';
import { MapSelectorProps, Coordinate, EcoRouteResponse } from '@/types/route';

// Define Leaflet types to avoid TypeScript errors
declare global {
    interface Window {
        L: any;
    }
}

const MapSelector: React.FC<MapSelectorProps> = ({ 
    onSelect, 
    route, 
    ecoRouteData, 
    showRouteType = 'both',
    fromCoord,
    toCoord 
}) => {
    const mapRef = useRef<HTMLDivElement>(null);
    const mapInstanceRef = useRef<any>(null);
    const [isMapReady, setIsMapReady] = useState(false);
    const [startPoint, setStartPoint] = useState<Coordinate | null>(null);
    const [endPoint, setEndPoint] = useState<Coordinate | null>(null);
    const startMarkerRef = useRef<any>(null);
    const endMarkerRef = useRef<any>(null);
    const ecoRouteLayerRef = useRef<any>(null);
    const googleRouteLayerRef = useRef<any>(null);

    // Initialize map
    useEffect(() => {
        if (!mapRef.current || isMapReady) return;

        const loadLeaflet = async () => {
            // Load Leaflet CSS
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
            document.head.appendChild(link);

            // Load Leaflet JS
            const script = document.createElement('script');
            script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
            script.onload = () => initializeMap();
            document.head.appendChild(script);
        };

        const initializeMap = () => {
            if (!window.L || !mapRef.current) return;

            const defaultCenter: [number, number] = [28.6139, 77.2090]; // Delhi
            const map = window.L.map(mapRef.current).setView(defaultCenter, 11);

            // Add OpenStreetMap tiles
            window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);

            // Add click event
            map.on('click', handleMapClick);

            mapInstanceRef.current = map;
            setIsMapReady(true);
        };

        loadLeaflet();

        return () => {
            if (mapInstanceRef.current) {
                mapInstanceRef.current.remove();
                mapInstanceRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (!fromCoord && !toCoord && (startPoint || endPoint)) {
            resetSelection();
        }
    }, [fromCoord, toCoord]);


    // Sync with parent component coordinates
    useEffect(() => {
        if (fromCoord && fromCoord !== startPoint) {
            setStartPoint(fromCoord);
            addStartMarker(fromCoord);
        }
    }, [fromCoord]);

    useEffect(() => {
        if (toCoord && toCoord !== endPoint) {
            setEndPoint(toCoord);
            addEndMarker(toCoord);
        }
    }, [toCoord]);

    // Handle route data updates
    useEffect(() => {
        if (!isMapReady || !ecoRouteData) return;

        // Clear existing routes
        clearRoutes();

        // Add eco route
        if (ecoRouteData.eco_route && (showRouteType === 'eco' || showRouteType === 'both')) {
            // eco_route is [number, number][]
            const ecoCoordinates = ecoRouteData.eco_route.map(([lat, lng]) => [lat, lng]);
            ecoRouteLayerRef.current = window.L.polyline(ecoCoordinates, {
                color: '#10B981',
                weight: 4,
                opacity: 0.8
            }).addTo(mapInstanceRef.current);
        }

        // Add Google route
        if (ecoRouteData.google_route && (showRouteType === 'google' || showRouteType === 'both')) {
            const googleCoordinates = ecoRouteData.google_route.map(([lat, lng]) => [lat, lng]);
            googleRouteLayerRef.current = window.L.polyline(googleCoordinates, {
                color: '#3B82F6',
                weight: 4,
                opacity: 0.8,
                dashArray: showRouteType === 'both' ? '10, 10' : ''
            }).addTo(mapInstanceRef.current);
        }

        // Fit map to show all routes
        const allCoordinates: [number, number][] = [];
        if (ecoRouteData.eco_route) {
            allCoordinates.push(...ecoRouteData.eco_route);
        }
        if (ecoRouteData.google_route) {
            allCoordinates.push(...ecoRouteData.google_route);
        }
        
        if (allCoordinates.length > 0) {
            const bounds = window.L.latLngBounds(allCoordinates);
            mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });
        }

    }, [ecoRouteData, showRouteType, isMapReady]);

    // Handle route type visibility changes
    useEffect(() => {
        if (!isMapReady || !ecoRouteData) return;

        // Toggle eco route visibility
        if (ecoRouteLayerRef.current) {
            if (showRouteType === 'eco' || showRouteType === 'both') {
                if (!mapInstanceRef.current.hasLayer(ecoRouteLayerRef.current)) {
                    ecoRouteLayerRef.current.addTo(mapInstanceRef.current);
                }
            } else {
                if (mapInstanceRef.current.hasLayer(ecoRouteLayerRef.current)) {
                    mapInstanceRef.current.removeLayer(ecoRouteLayerRef.current);
                }
            }
        }

        // Toggle Google route visibility
        if (googleRouteLayerRef.current) {
            if (showRouteType === 'google' || showRouteType === 'both') {
                if (!mapInstanceRef.current.hasLayer(googleRouteLayerRef.current)) {
                    googleRouteLayerRef.current.addTo(mapInstanceRef.current);
                }
                // Update dash pattern based on display mode
                googleRouteLayerRef.current.setStyle({
                    dashArray: showRouteType === 'both' ? '10, 10' : ''
                });
            } else {
                if (mapInstanceRef.current.hasLayer(googleRouteLayerRef.current)) {
                    mapInstanceRef.current.removeLayer(googleRouteLayerRef.current);
                }
            }
        }
    }, [showRouteType, isMapReady]);

    const handleMapClick = (e: any) => {
        const { lat, lng } = e.latlng;
        const coord: Coordinate = { lat, lng };

        // If neither point is set
        if (!startPoint && !endPoint) {
            setStartPoint(coord);
            addStartMarker(coord);
            return;
        }

        // If start is set and end is not
        if (startPoint && !endPoint) {
            setEndPoint(coord);
            addEndMarker(coord);
            onSelect(startPoint, coord); // startPoint is reliable here
            return;
        }

        // If both are set, reset
        resetSelection();
        setStartPoint(coord);
        addStartMarker(coord);
    };



    const addStartMarker = (coord: Coordinate) => {
        if (!mapInstanceRef.current || !window.L) return;

        // Remove existing start marker
        if (startMarkerRef.current) {
            mapInstanceRef.current.removeLayer(startMarkerRef.current);
        }

        const icon = window.L.divIcon({
            html: '<div style="background: #10B981; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">🚀</div>',
            iconSize: [30, 30],
            iconAnchor: [15, 15]
        });

        startMarkerRef.current = window.L.marker([coord.lat, coord.lng], { icon })
            .addTo(mapInstanceRef.current)
            .bindPopup("Start Point");
    };

    const addEndMarker = (coord: Coordinate) => {
        if (!mapInstanceRef.current || !window.L) return;

        // Remove existing end marker
        if (endMarkerRef.current) {
            mapInstanceRef.current.removeLayer(endMarkerRef.current);
        }

        const icon = window.L.divIcon({
            html: '<div style="background: #EF4444; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">🎯</div>',
            iconSize: [30, 30],
            iconAnchor: [15, 15]
        });

        endMarkerRef.current = window.L.marker([coord.lat, coord.lng], { icon })
            .addTo(mapInstanceRef.current)
            .bindPopup("End Point");
    };

    const clearRoutes = () => {
        if (ecoRouteLayerRef.current) {
            mapInstanceRef.current.removeLayer(ecoRouteLayerRef.current);
            ecoRouteLayerRef.current = null;
        }
        if (googleRouteLayerRef.current) {
            mapInstanceRef.current.removeLayer(googleRouteLayerRef.current);
            googleRouteLayerRef.current = null;
        }
    };

    const resetSelection = () => {
        setStartPoint(null);
        setEndPoint(null);
        
        // Remove markers
        if (startMarkerRef.current) {
            mapInstanceRef.current.removeLayer(startMarkerRef.current);
            startMarkerRef.current = null;
        }
        if (endMarkerRef.current) {
            mapInstanceRef.current.removeLayer(endMarkerRef.current);
            endMarkerRef.current = null;
        }
        
        // Clear routes
        clearRoutes();
    };

    const triggerRoute = () => {
        if (startPoint && endPoint) {
            onSelect(startPoint, endPoint);
        }
    };

    // Reset when parent clears coordinates
    useEffect(() => {
        if (!fromCoord && !toCoord) {
            resetSelection();
        }
    }, [fromCoord, toCoord]);

    return (
        <div className="relative w-full h-full">
            <div ref={mapRef} className="w-full h-full rounded-lg" />
            
            {/* Map Controls */}
            <div className="absolute top-4 left-4 z-[1000] bg-white rounded-lg shadow-lg p-3 border max-w-xs">
                <div className="text-sm font-medium mb-2 text-gray-700">
                    {!startPoint && "Click to select start location"}
                    {startPoint && !endPoint && "Click to select destination"}
                    {startPoint && endPoint && "Ready to calculate route"}
                </div>
                
                <div className="flex gap-2 flex-wrap">
                    <button
                        onClick={resetSelection}
                        className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600 transition-colors"
                    >
                        Reset
                    </button>
                    
                    {startPoint && endPoint && (
                        <button
                            onClick={triggerRoute}
                            className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 transition-colors"
                        >
                            Get Eco Route
                        </button>
                    )}
                </div>
            </div>

            {/* Route Legend */}
            {ecoRouteData && (
                <div className="absolute bottom-4 left-4 z-[1000] bg-white rounded-lg shadow-lg p-3 border">
                    <div className="text-sm font-medium mb-2 text-gray-700">Route Legend</div>
                    
                    {(showRouteType === 'eco' || showRouteType === 'both') && (
                        <div className="flex items-center gap-2 mb-1">
                            <div className="w-4 h-1 bg-green-500 rounded"></div>
                            <span className="text-xs text-gray-600">Eco Route</span>
                        </div>
                    )}
                    
                    {(showRouteType === 'google' || showRouteType === 'both') && (
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-1 bg-blue-500 rounded" 
                                     style={{ backgroundImage: showRouteType === 'both' ? 'repeating-linear-gradient(90deg, #3B82F6 0px, #3B82F6 5px, transparent 5px, transparent 10px)' : '' }}></div>
                            <span className="text-xs text-gray-600">Google Route</span>
                        </div>
                    )}
                </div>
            )}
            
            {/* Loading Indicator */}
            {!isMapReady && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
                    <div className="text-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
                        <p className="text-sm text-gray-600">Loading map...</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MapSelector;