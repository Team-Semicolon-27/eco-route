"use client";
import { useState, useEffect } from "react";
import MapSelector from "@/components/MapSelector";
import { Coordinate, EcoRouteResponse, Vehicle } from "@/types/route";

export default function DashboardPage() {
  const [route, setRoute] = useState<Coordinate[] | null>(null);
  const [ecoRouteData, setEcoRouteData] = useState<EcoRouteResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [selectedVehicle, setSelectedVehicle] = useState<string>("pass. car");
  const [showRouteType, setShowRouteType] = useState<'eco' | 'google' | 'both'>('both');
  const [fromCoord, setFromCoord] = useState<Coordinate | null>(null);
  const [toCoord, setToCoord] = useState<Coordinate | null>(null);

  // Fetch available vehicles on component mount
  useEffect(() => {
    const fetchVehicles = async () => {
      try {
        const res = await fetch("/api/vehicles");
        if (res.ok) {
          const data = await res.json();
          setVehicles(data.vehicles);
        } else {
          throw new Error("Failed to fetch vehicles");
        }
      } catch (err) {
        console.error("Failed to fetch vehicles:", err);
        // Fallback vehicles
        setVehicles([
          { id: "pass. car", name: "Passenger Car" },
          { id: "motorcycle", name: "Motorcycle" },
          { id: "LCV", name: "Light Commercial Vehicle" },
          { id: "HGV", name: "Heavy Goods Vehicle" },
          { id: "coach", name: "Coach" },
          { id: "urban bus", name: "Urban Bus" }
        ]);
      }
    };
    fetchVehicles();
  }, []);

  const handleRouteFetch = async (from: Coordinate, to: Coordinate) => {
    setError("");
    setLoading(true);
    
    // Update coordinates first
    setFromCoord(from);
    setToCoord(to);

    try {
      // Call EcoRoute API
      const ecoRouteRequest = {
        origin: [from.lat, from.lng],
        destination: [to.lat, to.lng],
        vehicle: selectedVehicle
      };

      const res = await fetch("/api/ecoroute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ecoRouteRequest),
      });

      if (!res.ok) {
        const errorData = await res.text();
        throw new Error(`API Error: ${res.status} - ${errorData}`);
      }
      
      const data: EcoRouteResponse = await res.json();
      setEcoRouteData(data);
      
      // Clear the old route since we're now using ecoRouteData
      setRoute(null);

    } catch (err) {
      console.error("Route fetch error:", err);
      setError(`Unable to fetch eco-optimized route: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setEcoRouteData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleVehicleChange = (vehicle: string) => {
    setSelectedVehicle(vehicle);
    // Re-fetch route if coordinates are available
    if (fromCoord && toCoord) {
      handleRouteFetch(fromCoord, toCoord);
    }
  };

  const handleRouteTypeChange = (type: 'eco' | 'google' | 'both') => {
    setShowRouteType(type);
  };

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      setLoading(true);
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const currentPos: Coordinate = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
          };
          setFromCoord(currentPos);
          setError("");
          setLoading(false);
        },
        (error) => {
          console.error("Error getting location:", error);
          setError("Unable to get current location. Please select manually.");
          setLoading(false);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 60000
        }
      );
    } else {
      setError("Geolocation is not supported by this browser.");
    }
  };

  const clearRoute = () => {
    setRoute(null);
    setEcoRouteData(null);
    setFromCoord(null);
    setToCoord(null);
    setError("");
  };


  const formatNumber = (num: number, decimals: number = 2): string => {
    return num.toFixed(decimals);
  };

  const formatTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-950 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Sidebar - Controls */}
          <div className="lg:col-span-1 space-y-6 max-h-screen overflow-y-auto">
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 p-6 rounded-2xl shadow-2xl">
              <h1 className="text-3xl font-bold text-center mb-2 bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                🌱 EcoRoute AI
              </h1>
              <p className="text-sm text-white/70 text-center mb-6">
                Smart route planning for eco-friendly travel
              </p>

              {/* Vehicle Selection */}
              <div className="mb-6">
                <label className="block text-sm font-semibold mb-2 text-white/90">
                  Vehicle Type:
                </label>
                <select
                  value={selectedVehicle}
                  onChange={(e) => handleVehicleChange(e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                  disabled={loading}
                >
                  {vehicles.map((vehicle) => (
                    <option key={vehicle.id} value={vehicle.id} className="bg-gray-800 text-white">
                      {vehicle.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Action Buttons */}
              <div className="space-y-3 mb-6">
                <button
                  onClick={getCurrentLocation}
                  disabled={loading}
                  className="w-full px-4 py-3 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 disabled:hover:scale-100 shadow-lg"
                >
                  {loading ? "Getting location..." : "📍 Use Current Location as Start"}
                </button>
                
                {(fromCoord || toCoord || ecoRouteData) && (
                  <button
                    onClick={clearRoute}
                    disabled={loading}
                    className="w-full px-4 py-3 bg-gradient-to-r from-gray-500 to-gray-600 hover:from-gray-600 hover:to-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 disabled:hover:scale-100 shadow-lg"
                  >
                    🗑️ Clear Route
                  </button>
                )}
              </div>

              {/* Route Type Selection */}
              {ecoRouteData && (
                <div className="mb-6">
                  <label className="block text-sm font-semibold mb-2 text-white/90">
                    Display Route:
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleRouteTypeChange('eco')}
                      className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                        showRouteType === 'eco'
                          ? 'bg-green-500 text-white shadow-lg'
                          : 'bg-white/10 text-white/70 hover:bg-white/20'
                      }`}
                    >
                      🌱 Eco
                    </button>
                    <button
                      onClick={() => handleRouteTypeChange('google')}
                      className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                        showRouteType === 'google'
                          ? 'bg-blue-500 text-white shadow-lg'
                          : 'bg-white/10 text-white/70 hover:bg-white/20'
                      }`}
                    >
                      🚗 Google
                    </button>
                    <button
                      onClick={() => handleRouteTypeChange('both')}
                      className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                        showRouteType === 'both'
                          ? 'bg-purple-500 text-white shadow-lg'
                          : 'bg-white/10 text-white/70 hover:bg-white/20'
                      }`}
                    >
                      📍 Both
                    </button>
                  </div>
                </div>
              )}

              {/* Route Statistics */}
              {ecoRouteData && (
                <div className="bg-green-500/20 border border-green-400/30 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold text-green-300 mb-3">Route Comparison</h4>
                  
                  {/* Eco Route Stats */}
                  <div className="mb-3">
                    <div className="text-sm text-green-200 font-medium mb-1">🌱 Eco Route</div>
                    <div className="text-xs text-green-100 grid grid-cols-2 gap-2">
                      <div>Distance: {formatNumber(ecoRouteData.eco_stats.distance_km / 1000, 1)} km</div>
                      <div>Duration: {formatTime(ecoRouteData.eco_stats.time_minutes / 60)}</div>
                      <div>CO₂: {formatNumber(ecoRouteData.eco_stats.co2_kg, 2)} kg</div>
                    </div>
                  </div>

                  {/* Google Route Stats */}
                  <div className="mb-3">
                    <div className="text-sm text-blue-200 font-medium mb-1">🚗 Google Route</div>
                    <div className="text-xs text-blue-100 grid grid-cols-2 gap-2">
                      <div>Distance: {formatNumber(ecoRouteData.google_stats.distance_km / 1000, 1)} km</div>
                      <div>Duration: {formatTime(ecoRouteData.google_stats.time_minutes / 60)}</div>
                      <div>CO₂: {formatNumber(ecoRouteData.google_stats.co2_kg, 2)} kg</div>
                    </div>
                  </div>

                  {/* Savings */}
                </div>
              )}

              {/* Current Selection Display */}
              {(fromCoord || toCoord) && (
                <div className="bg-blue-500/20 border border-blue-400/30 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold text-blue-300 mb-2">Selected Points:</h4>
                  {fromCoord && (
                    <div className="text-sm text-blue-200 mb-1">
                      🚀 Start: {fromCoord.lat.toFixed(4)}, {fromCoord.lng.toFixed(4)}
                    </div>
                  )}
                  {toCoord && (
                    <div className="text-sm text-blue-200">
                      🎯 End: {toCoord.lat.toFixed(4)}, {toCoord.lng.toFixed(4)}
                    </div>
                  )}
                </div>
              )}

              {/* Instructions */}
              <div className="bg-blue-500/20 border border-blue-400/30 rounded-lg p-4 mb-4">
                <p className="text-sm text-blue-200">
                  {!fromCoord && !toCoord && "Click on the map to select your starting location first, then select your destination."}
                  {fromCoord && !toCoord && "Great! Now click on the map to select your destination."}
                  {fromCoord && toCoord && "Perfect! Both locations selected. Route will be calculated automatically."}
                </p>
              </div>

              {/* Status Messages */}
              <div className="text-center">
                {loading && (
                  <p className="text-sm text-yellow-300 animate-pulse">Calculating eco-optimized route...</p>
                )}
                {error && (
                  <p className="text-sm text-red-400">{error}</p>
                )}
                {!loading && !error && ecoRouteData && (
                  <p className="text-sm text-green-300">EcoRoute calculated successfully! View it on the map.</p>
                )}
              </div>
            </div>
          </div>

          {/* Right Section - Map Viewer */}
          <div className="lg:col-span-2">
            <div className="w-full h-[calc(100vh-4rem)] rounded-xl overflow-hidden shadow-2xl border border-white/20">
              <MapSelector
                onSelect={handleRouteFetch}
                route={route}
                ecoRouteData={ecoRouteData}
                showRouteType={showRouteType}
                fromCoord={fromCoord}
                toCoord={toCoord}
              />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}