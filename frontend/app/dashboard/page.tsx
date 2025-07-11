"use client";

import React, { useState, useEffect, useRef } from "react";
import { GeoapifyContext } from "@geoapify/react-geocoder-autocomplete";
import type { LatLngTuple } from "leaflet";

import MapControls from "@/components/MapControls";
import MapDisplay from "@/components/MapDisplay";

interface EcoStats {
  distance_km: number;
  time_minutes: number;
  time_minutes_google_estimated: number;
  co2_kg: number;
}

interface GoogleStats {
  distance_km: number;
  time_minutes: number;
  co2_kg: number;
}

interface Comparison {
  co2_savings_kg: number;
  co2_savings_percent: number;
  time_difference_minutes: number;
}

export default function Page() {
  const [source, setSource] = useState<any>(null);
  const [destination, setDestination] = useState<any>(null);
  const [vehicle, setVehicle] = useState("motorcycle");
  const [loading, setLoading] = useState(false);

  const [ecoRoute, setEcoRoute] = useState<LatLngTuple[]>([]);
  const [googleRoute, setGoogleRoute] = useState<LatLngTuple[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  // New state for stats
  const [ecoStats, setEcoStats] = useState<EcoStats | null>(null);
  const [googleStats, setGoogleStats] = useState<GoogleStats | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);

  const handleRouteFind = () => {
    if (!source || !destination) {
      alert("Please select both source and destination.");
      return;
    }
    eventSourceRef.current?.close();
    setLoading(true);
    setLogs([]);
    setEcoRoute([]);
    setGoogleRoute([]);

    // Reset stats
    setEcoStats(null);
    setGoogleStats(null);
    setComparison(null);

    const origin = `${source.properties.lat},${source.properties.lon}`;
    const dest = `${destination.properties.lat},${destination.properties.lon}`;
    const url = `http://127.0.0.1:8000/route/stream?origin=${origin}&destination=${dest}&vehicle=${vehicle}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const { type, message, data } = JSON.parse(event.data);
      if (type === "log") {
        setLogs((l) => [...l, message]);
      } else if (type === "result") {
        setEcoRoute(data.eco_route);
        setGoogleRoute(data.google_route);

        // Set the stats data
        if (data.eco_stats) {
          setEcoStats(data.eco_stats);
        }
        if (data.google_stats) {
          setGoogleStats(data.google_stats);
        }
        if (data.comparison) {
          setComparison(data.comparison);
        }

        setLoading(false);
        es.close();
      } else if (type === "error") {
        setLogs((l) => [...l, `[ERROR] ${message}`]);
        alert(`Error: ${message}`);
        setLoading(false);
        es.close();
      }
    };

    es.onerror = (err) => {
      console.error("EventSource failed:", err);
      setLogs((l) => [...l, "[ERROR] Connection to server lost."]);
      setLoading(false);
      es.close();
    };
  };

  const handleGoogleMapOpen = () => {
    if (ecoRoute.length < 2) return;
    let pts = ecoRoute;
    const max = 25;
    if (pts.length > max) {
      const step = Math.floor(pts.length / (max - 1));
      pts = pts.filter((_, i) => i % step === 0);
      if (pts[pts.length - 1] !== ecoRoute[ecoRoute.length - 1]) {
        pts.push(ecoRoute[ecoRoute.length - 1]);
      }
    }
    const origin = `${pts[0][0]},${pts[0][1]}`;
    const dest = `${pts[pts.length - 1][0]},${pts[pts.length - 1][1]}`;
    const waypoints = pts
      .slice(1, -1)
      .map(([lat, lon]) => `${lat},${lon}`)
      .join("|");
    const url = `https://www.google.com/maps/dir/?api=1&origin=${origin}&destination=${dest}&waypoints=${waypoints}`;
    window.open(url, "_blank");
  };

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  return (
    <GeoapifyContext apiKey={process.env.NEXT_PUBLIC_GEOAPIFY_API_KEY}>
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ backgroundColor: "#365241" }}
      >
        <div
          className="rounded-2xl w-[90vw] h-[95vh] p-8"
          style={{ backgroundColor: "#F0EDD1", color: "#243931" }}
        >
          <h1 className="text-5xl font-bold mb-4">Dashboard</h1>
          <div className="flex justify-between items-start gap-4 h-[calc(100%-60px)]">
            <div className="flex-[1] h-full">
              <MapControls
                source={source}
                setSource={setSource}
                destination={destination}
                setDestination={setDestination}
                vehicle={vehicle}
                setVehicle={setVehicle}
                loading={loading}
                routeReady={ecoRoute.length > 0}
                onFind={handleRouteFind}
                onOpenGoogleMaps={handleGoogleMapOpen}
              />
            </div>
            <div className="flex-[2] h-full">
              <MapDisplay
                source={source}
                destination={destination}
                loading={loading}
                ecoRoute={ecoRoute}
                googleRoute={googleRoute}
                logs={logs}
                ecoStats={ecoStats}
                googleStats={googleStats}
                comparison={comparison}
              />
            </div>
          </div>
        </div>
      </div>
    </GeoapifyContext>
  );
}
