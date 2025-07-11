"use client";

import React, { useState, useEffect, useRef } from "react";

// Defines the types for the component's props
interface LogDisplayProps {
  loading: boolean;
  logs: string[];
}

const LogDisplay: React.FC<LogDisplayProps> = ({ loading, logs = [] }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false); // start closed
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the bottom when new logs are added
  useEffect(() => {
    if (isOpen && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, isOpen]);

  const latestLog =
    logs.length > 0 ? logs[logs.length - 1] : "Awaiting request...";

  return (
    <div className="rounded-lg border border-white/20 bg-white/10 backdrop-blur-lg shadow-lg transition-all duration-300 w-full text-white">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full p-3 text-left flex justify-between items-center"
      >
        <div className="flex items-center min-w-0">
          {loading && (
            <svg
              className="animate-spin h-5 w-5 text-white flex-shrink-0"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          )}
          {!isOpen && (
            <span className="ml-3 font-mono text-xs truncate">
              {loading ? "Processing..." : latestLog}
            </span>
          )}
        </div>
        <span
          className={`transform transition-transform duration-200 ${
            isOpen ? "rotate-180" : ""
          }`}
        >
          ▼
        </span>
      </button>

      {/* Animated Collapsible Content */}
      <div
        ref={logContainerRef}
        className={`transition-all duration-300 overflow-hidden bg-black/20 border-t border-white/20 font-mono text-xs leading-relaxed ${
          isOpen
            ? "max-h-64 opacity-100 py-4 px-4"
            : "max-h-0 opacity-0 py-0 px-4"
        }`}
      >
        {logs.length > 0 ? (
          logs.map((log, index) => (
            <p
              key={index}
              className="whitespace-pre-wrap break-words text-white text-xs"
            >
              {log}
            </p>
          ))
        ) : (
          <p className="text-white/50 italic">
            Sending your request to our server
          </p>
        )}
      </div>
    </div>
  );
};

export default LogDisplay;
