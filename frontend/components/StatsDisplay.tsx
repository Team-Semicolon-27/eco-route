"use client";

import React, { useState, useEffect } from "react";
import { Leaf, Clock, MapPin, TrendingDown } from "lucide-react";

interface StatsDisplayProps {
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
  isVisible: boolean;
}

const AnimatedNumber: React.FC<{
  value: number;
  duration?: number;
  suffix?: string;
  prefix?: string;
}> = ({ value, duration = 1000, suffix = "", prefix = "" }) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let startTime: number;
    let animationFrame: number;

    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(value * easeOut);
      if (progress < 1) animationFrame = requestAnimationFrame(animate);
    };

    animationFrame = requestAnimationFrame(animate);

    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
    };
  }, [value, duration]);

  return (
    <span>
      {prefix}
      {displayValue}
      {suffix}
    </span>
  );
};

const CombinedStatCard: React.FC<{
  icon: React.ReactNode;
  title: string;
  ecoValue: number;
  googleValue: number;
  unit: string;
  delay?: number;
}> = ({ icon, title, ecoValue, googleValue, unit, delay = 0 }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <div
      className={`backdrop-blur-sm rounded-lg p-2 shadow-lg transition-all duration-500 transform ${
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      }`}
      style={{
        backgroundColor: "rgba(35, 56, 48, 0.9)",
        transitionDelay: `${delay}ms`,
      }}
    >
      <div className="flex items-center gap-1 mb-1">
        <div
          className="p-1 rounded-full"
          style={{ backgroundColor: "rgba(240, 237, 209, 0.2)" }}
        >
          {icon}
        </div>
        <span
          className="text-xs font-medium truncate"
          style={{ color: "#F0EDD1" }}
        >
          {title}
        </span>
      </div>

      <div className="space-y-1">
        {isVisible ? (
          <>
            <div className="text-sm font-bold" style={{ color: "#F0EDD1" }}>
              <AnimatedNumber value={ecoValue} suffix={unit} duration={800} />
            </div>
            <div
              className="text-xs"
              style={{ color: "rgba(240, 237, 209, 0.7)" }}
            >
              G:{" "}
              <AnimatedNumber
                value={googleValue}
                suffix={unit}
                duration={800}
              />
            </div>
          </>
        ) : (
          <>
            <div className="text-sm font-bold" style={{ color: "#F0EDD1" }}>
              0{unit}
            </div>
            <div
              className="text-xs"
              style={{ color: "rgba(240, 237, 209, 0.7)" }}
            >
              G: 0{unit}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const ComparisonCard: React.FC<{
  title: string;
  value: number;
  unit: string;
  isPositive: boolean;
  secondaryValue?: number;
  secondaryUnit?: string;
  delay?: number;
}> = ({
  title,
  value,
  unit,
  isPositive,
  secondaryValue,
  secondaryUnit,
  delay = 0,
}) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <div
      className={`backdrop-blur-sm rounded-lg p-3 shadow-lg transition-all duration-500 transform ${
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      }`}
      style={{
        backgroundColor: "rgba(35, 56, 48, 0.9)",
        transitionDelay: `${delay}ms`,
      }}
    >
      <div className="flex items-center gap-2 mb-1">
        <div
          className={`p-1 rounded-full ${
            isPositive ? "bg-green-100" : "bg-red-100"
          }`}
        >
          <TrendingDown
            className={`w-3 h-3 ${
              isPositive ? "text-green-600" : "text-red-600"
            }`}
          />
        </div>
        <span className="text-xs font-medium" style={{ color: "#F0EDD1" }}>
          {title}
        </span>
      </div>
      <div
        className={`text-lg font-bold ${
          isPositive ? "text-green-600" : "text-red-600"
        }`}
      >
        {isVisible ? (
          <span>
            {isPositive ? "↓" : "↑"}
            <AnimatedNumber
              value={Math.abs(value)}
              suffix={unit}
              duration={800}
            />
            {secondaryValue !== undefined && secondaryUnit && (
              <span
                className="text-sm ml-2 font-normal"
                style={{ color: "#F0EDD1" }}
              >
                (
                <AnimatedNumber
                  value={Math.abs(secondaryValue)}
                  suffix={secondaryUnit}
                  duration={800}
                />
                )
              </span>
            )}
          </span>
        ) : (
          `${isPositive ? "↓" : "↑"}0${unit}${
            secondaryValue !== undefined && secondaryUnit
              ? ` (${secondaryValue}${secondaryUnit})`
              : ""
          }`
        )}
      </div>
    </div>
  );
};

export default function StatsDisplay({
  ecoStats,
  googleStats,
  comparison,
  isVisible,
}: StatsDisplayProps) {
  const [showStats, setShowStats] = useState(false);

  useEffect(() => {
    if (isVisible && ecoStats && googleStats && comparison) {
      const timer = setTimeout(() => setShowStats(true), 200);
      return () => clearTimeout(timer);
    } else {
      setShowStats(false);
    }
  }, [isVisible, ecoStats, googleStats, comparison]);

  if (!showStats || !ecoStats || !googleStats || !comparison) {
    return null;
  }

  return (
    <div className="absolute top-4 right-4 z-[1000] max-w-xs">
      <div
        className={`transition-all duration-700 transform ${
          showStats ? "opacity-100 translate-x-0" : "opacity-0 translate-x-8"
        }`}
      >
        <div
          className="backdrop-blur-sm rounded-xl p-4 overflow-y-auto max-h-[calc(100vh-4rem)]"
          style={{ backgroundColor: "rgba(35, 56, 48, 0.8)" }}
        >
          <h3
            className="font-bold text-lg mb-3 flex items-center gap-2"
            style={{ color: "#F0EDD1" }}
          >
            <Leaf className="w-5 h-5" />
            Route Comparison
          </h3>

          <div className="space-y-4 pr-1">
            <div className="space-y-2">
              <h4
                className="font-semibold text-sm"
                style={{ color: "rgba(240, 237, 209, 0.9)" }}
              >
                Savings
              </h4>
              <div className="flex gap-2">
                <ComparisonCard
                  title="CO₂ Savings"
                  value={comparison.co2_savings_kg}
                  unit=" kg"
                  secondaryValue={comparison.co2_savings_percent}
                  secondaryUnit="%"
                  isPositive={comparison.co2_savings_kg > 0}
                  delay={700}
                />
                <ComparisonCard
                  title="Time Difference"
                  value={comparison.time_difference_minutes}
                  unit=" min"
                  isPositive={comparison.time_difference_minutes < 0}
                  delay={900}
                />
              </div>
            </div>

            <div className="space-y-2">
              <h4
                className="font-semibold text-sm"
                style={{ color: "rgba(240, 237, 209, 0.9)" }}
              >

              </h4>
              <div className="grid grid-cols-3 gap-1">
                <CombinedStatCard
                  icon={<MapPin className="w-3 h-3 text-green-600" />}
                  title="Distance"
                  ecoValue={ecoStats.distance_km}
                  googleValue={googleStats.distance_km}
                  unit=" km"
                  delay={100}
                />
                <CombinedStatCard
                  icon={<Clock className="w-3 h-3 text-blue-600" />}
                  title="Time"
                  ecoValue={ecoStats.time_minutes}
                  googleValue={googleStats.time_minutes}
                  unit=" min"
                  delay={200}
                />
                <CombinedStatCard
                  icon={<Leaf className="w-3 h-3 text-green-600" />}
                  title="CO₂"
                  ecoValue={ecoStats.co2_kg}
                  googleValue={googleStats.co2_kg}
                  unit=" kg"
                  delay={300}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
