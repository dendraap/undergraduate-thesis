import { useEffect, useState } from "react";
import MainGrid from "../components/MainGrid";
import { Stack } from "@mui/material";

interface TFTPageProps {
    pollutantKey: string;
}

export default function TFTPage({ pollutantKey }: TFTPageProps) {
    const [weatherData, setWeatherData] = useState<any>(null);
    const [pollutantData, setPollutantData] = useState<any>(null)
    const [loading, setLoading] = useState(true)
   
    useEffect(() => {
        const fetchData = async () => {
            const res = await fetch("http://127.0.0.1:8000/forecast/?model=tft");
            const data = await res.json();

            setPollutantData(data.pollutants);
            setWeatherData(data.weather);
            setLoading(false);
        };

        fetchData();
    }, []);
    
    if (loading) return <>Loading...</>;
    if (!pollutantData && !weatherData) return null;

    return (
        <Stack spacing={3} sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
            <MainGrid
                pollutantKey={pollutantKey}
                pollutantData={pollutantData}
                weatherData={weatherData}
            />
        </Stack>
    );
}
