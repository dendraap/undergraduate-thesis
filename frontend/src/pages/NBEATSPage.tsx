import { useEffect, useState } from "react";
import Header from "../components/Header";
import MainGrid from "../components/MainGrid";
import PollutanSelector from "../components/PollutanSelector";

export default function TFTPage() {
    const [pollutant, setPollutant] = useState("pm25")
    const [pollutantData, setPollutantData] = useState<any>(null)
    const [weatherData, setWeatherData] = useState<any>(null)
    const [loading, setLoading] = useState(true)

    const fecthData = async () => {
        const response = await fetch("http://127.0.0.1:8000/forecast/?model=tft")
        const data = await response.json()
        // console.log(data)
        setPollutantData(data.pollutants)
        setWeatherData(data.weather)
        setLoading(false)
    }

    useEffect(() => {
        fecthData()
    }, [])
    
    if (loading) return <>Loading...</>;
    if (!pollutantData && !weatherData) return null;

    return (
        <>
            <MainGrid
                pollutantKey={pollutant}
                pollutantData={pollutantData[pollutant]}
                weatherData={weatherData}
            />
        </>
    );
}
