"use client";

import React, { useRef, useEffect } from "react";
import L, { Marker, LatLngTuple } from "leaflet";
import "leaflet/dist/leaflet.css";
import LoadingOverlay from "./LoadingOverlay";
import StatsDisplay from "./StatsDisplay";

interface MapDisplayProps {
  source: any;
  destination: any;
  loading: boolean;
  ecoRoute: LatLngTuple[];
  googleRoute: LatLngTuple[];
  logs: string[];
  ecoStats: {
    distance_km: number;
    time_minutes: number;
    time_minutes_google_estimated: number;
    co2_kg: number;
  } | null;
  googleStats: {
    distance_km: number;
    time_minutes: number;
    co2_kg: number;
  } | null;
  comparison: {
    co2_savings_kg: number;
    co2_savings_percent: number;
    time_difference_minutes: number;
  } | null;
}

const createCustomIcon = (
  iconUrl: string,
  size: [number, number] = [32, 32],
  anchor: [number, number] = [16, 32]
) => {
  return L.icon({
    iconUrl,
    iconSize: size,
    iconAnchor: anchor,
    popupAnchor: [0, -32] as [number, number],
    shadowUrl:
      "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
    shadowSize: [41, 41] as [number, number],
    shadowAnchor: [12, 41] as [number, number],
  });
};

const sourceIcon = createCustomIcon("/marker1.png");
const destinationIcon = createCustomIcon("/marker2.png");

export default function MapDisplay({
  source,
  destination,
  loading,
  ecoRoute,
  googleRoute,
  logs,
  ecoStats,
  googleStats,
  comparison,
}: MapDisplayProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const sourceMarkerRef = useRef<Marker | null>(null);
  const destinationMarkerRef = useRef<Marker | null>(null);
  const ecoPolylineRef = useRef<L.Polyline | null>(null);
  const googlePolylineRef = useRef<L.Polyline | null>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    const map = L.map(mapRef.current).setView([28.6139, 77.209], 5);
    mapInstanceRef.current = map;

    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}.png",
      {
        attribution:
          '© <a href="https://carto.com/">CARTO</a> | © OpenStreetMap contributors',
      }
    ).addTo(map);

    return () => {
      if (sourceMarkerRef.current) sourceMarkerRef.current.remove();
      if (destinationMarkerRef.current) destinationMarkerRef.current.remove();
      if (ecoPolylineRef.current) map.removeLayer(ecoPolylineRef.current);
      if (googlePolylineRef.current) map.removeLayer(googlePolylineRef.current);
      map.remove();
    };
  }, []);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (sourceMarkerRef.current) {
      sourceMarkerRef.current.remove();
      sourceMarkerRef.current = null;
    }

    if (source?.properties?.lat && source?.properties?.lon) {
      const marker = L.marker([source.properties.lat, source.properties.lon], {
        title: source.properties.formatted || "Source",
        icon: sourceIcon,
      }).addTo(map);
      sourceMarkerRef.current = marker;

      if (destination?.properties?.lat && destination?.properties?.lon) {
        const bounds = L.latLngBounds(
          [source.properties.lat, source.properties.lon],
          [destination.properties.lat, destination.properties.lon]
        );
        map.fitBounds(bounds, { padding: [50, 50] });
      } else {
        map.setView([source.properties.lat, source.properties.lon], 10);
      }
    }
  }, [source, destination]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (destinationMarkerRef.current) {
      destinationMarkerRef.current.remove();
      destinationMarkerRef.current = null;
    }

    if (destination?.properties?.lat && destination?.properties?.lon) {
      const marker = L.marker(
        [destination.properties.lat, destination.properties.lon],
        {
          title: destination.properties.formatted || "Destination",
          icon: destinationIcon,
        }
      ).addTo(map);
      destinationMarkerRef.current = marker;

      if (source?.properties?.lat && source?.properties?.lon) {
        const bounds = L.latLngBounds(
          [source.properties.lat, source.properties.lon],
          [destination.properties.lat, destination.properties.lon]
        );
        map.fitBounds(bounds, { padding: [50, 50] });
      } else {
        map.setView(
          [destination.properties.lat, destination.properties.lon],
          10
        );
      }
    }
  }, [destination, source]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (ecoPolylineRef.current) {
      map.removeLayer(ecoPolylineRef.current);
      ecoPolylineRef.current = null;
    }

    if (googlePolylineRef.current) {
      map.removeLayer(googlePolylineRef.current);
      googlePolylineRef.current = null;
    }

    if (ecoRoute && ecoRoute.length > 0) {
      ecoPolylineRef.current = L.polyline(ecoRoute, {
        color: "#233830",
        weight: 5,
        opacity: 0.9,
      }).addTo(map);
    }

    if (googleRoute && googleRoute.length > 0) {
      googlePolylineRef.current = L.polyline(googleRoute, {
        color: "#4f4f4f",
        weight: 4,
        opacity: 0.7,
      }).addTo(map);
    }
  }, [ecoRoute, googleRoute]);

  const shouldShowStats =
    !loading &&
    ecoStats !== null &&
    googleStats !== null &&
    comparison !== null &&
    ecoRoute.length > 0;

  return (
    <div
      className="relative rounded-2xl overflow-hidden"
      style={{
        backgroundColor: "#233830",
        width: "60vw",
        height: "75vh",
        zIndex: 0, // create stacking context
      }}
    >
      <div ref={mapRef} style={{ width: "100%", height: "100%", zIndex: 0 }} />
      <LoadingOverlay isLoading={loading} logs={logs} />
      <StatsDisplay
        ecoStats={ecoStats}
        googleStats={googleStats}
        comparison={comparison}
        isVisible={shouldShowStats}
      />
    </div>
  );
}
