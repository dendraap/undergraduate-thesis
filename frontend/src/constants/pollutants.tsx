export const POLLUTANTS = [
    { key: "pm25", label: "PM2.5" },
    { key: "pm10", label: "PM10" },
    { key: "so2",  label: "SO₂"  },
    { key: "no2",  label: "NO₂"  },
    { key: "o3",   label: "O₃"   },
    { key: "co",   label: "CO"   },
] as const;

export const POLLUTANT_LABEL_MAP: Record<string, string> =
    Object.fromEntries(POLLUTANTS.map(p => [p.key, p.label]));