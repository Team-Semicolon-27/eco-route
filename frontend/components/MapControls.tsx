// components/MapControls.tsx
"use client";

import React from "react";
import { GeoapifyGeocoderAutocomplete } from "@geoapify/react-geocoder-autocomplete";
import "@geoapify/geocoder-autocomplete/styles/minimal.css";
import AnimatedButton from "@/components/UI elements/FindButton";
import GoogleMapButton from "@/components/UI elements/GMapButtom";

interface MapControlsProps {
  source: any;
  setSource: (place: any) => void;
  destination: any;
  setDestination: (place: any) => void;
  vehicle: string;
  setVehicle: (v: string) => void;
  loading: boolean;
  routeReady: boolean;
  onFind: () => void;
  onOpenGoogleMaps: () => void;
}

export default function MapControls({
  source,
  setSource,
  destination,
  setDestination,
  vehicle,
  setVehicle,
  loading,
  routeReady,
  onFind,
  onOpenGoogleMaps,
}: MapControlsProps) {
  return (
    <div
      className="rounded-2xl p-6 flex flex-col"
      style={{
        backgroundColor: "#233830",
        width: "23vw",
        height: "75vh",
        color: "#F0EDD1",
      }}
    >
      <div className="flex flex-col gap-4">
        {/* Source */}
        <div>
          <label className="block mb-1 text-sm font-medium">Source</label>
          <div
            className="w-full rounded-2xl px-4 py-2"
            style={{
              backgroundColor: "rgba(255,255,255,0.1)",
              backdropFilter: "blur(10px)",
              border: "1px solid rgba(255,255,255,0.2)",
              position: "relative",
              zIndex: 99,
            }}
          >
            <GeoapifyGeocoderAutocomplete
              placeholder="Enter source"
              placeSelect={(place) => setSource(place)}
            />
          </div>
        </div>

        {/* Destination */}
        <div>
          <label className="block mb-1 text-sm font-medium">Destination</label>
          <div
            className="w-full rounded-2xl px-4 py-2"
            style={{
              backgroundColor: "rgba(255,255,255,0.1)",
              backdropFilter: "blur(10px)",
              border: "1px solid rgba(255,255,255,0.2)",
              position: "relative",
              zIndex: 9,
            }}
          >
            <GeoapifyGeocoderAutocomplete
              placeholder="Enter destination"
              placeSelect={(place) => setDestination(place)}
            />
          </div>
        </div>

        {/* Vehicle */}
        <div>
          <label className="block mb-1 text-sm font-medium">Vehicle Type</label>
          <select
            value={vehicle}
            onChange={(e) => setVehicle(e.target.value)}
            className="w-full rounded-2xl px-4 py-2"
            style={{
              backgroundColor: "#ACC08D",
              color: "#233830",
              border: "1px solid rgba(255,255,255,0.2)",
            }}
          >
            <option value="motorcycle">Motorcycle</option>
            <option value="pass. car">Passenger Car</option>
            <option value="LCV">Light Commercial Vehicle</option>
            <option value="coach">Coach</option>
            <option value="HGV">Heavy Goods Vehicle</option>
            <option value="urban bus">Urban Bus</option>
          </select>
        </div>

        {/* Find Route */}
        <div className="mt-2">
          <AnimatedButton
            disabled={!source || !destination || loading}
            onClick={onFind}
          />
        </div>

        {/* Open in Google Maps */}
        {routeReady && (
          <div className="mt-2">
            <GoogleMapButton disabled={loading} onClick={onOpenGoogleMaps} />
          </div>
        )}
      </div>
    </div>
  );
}
