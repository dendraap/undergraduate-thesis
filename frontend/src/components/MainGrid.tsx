import CardChart from "./CardChart";
import CardMini from "./CardMini";
import CardNotify from "./CardNotify";
import { Grid, Stack, Typography } from "@mui/material";
import { POLLUTANT_LABEL_MAP } from "../constants/pollutants";


interface MainGridProps {
    pollutantKey: string;
    pollutantData: any;
    weatherData: any;
}

export default function MainGrid({ pollutantKey, pollutantData, weatherData }: MainGridProps) {
    const ispu = pollutantData[pollutantKey].ispu;
    const concentration = pollutantData[pollutantKey].concentration;
    const overview = pollutantData[pollutantKey].overview;
    // const recommendations = pollutantData[pollutantKey].recommendations;
    const pollutantLabel = POLLUTANT_LABEL_MAP[pollutantKey] ?? pollutantKey;

    console.log('MainGrid', concentration)
    return (
        <Stack spacing={3} sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
            {/* cards */}
            <Typography component="h2" variant="h6" sx={{ mb: 2 }}>
                Overview
            </Typography>

            {/* first row */}
            <Grid
                container
                spacing={3}
                height={'auto'}

            >
                <Grid size={{xs: 12, sm: 6, md: 3}}>
                     <CardMini
                            title="ISPU"
                            latestValue={ispu.value}
                            deltaPercent={ispu.delta_percent}
                            series={ispu.sparkline}
                            window={ispu.window}
                            category={ispu.category}
                            showDot
                        />
                </Grid>
                <Grid size={{xs: 12, sm: 6, md: 3}}>
                     <CardMini
                            title="Concentration"
                            latestValue={concentration.value}
                            deltaPercent={concentration.delta_percent}
                            series={concentration.sparkline}
                            unit={concentration.unit}
                            window={concentration.window}
                        />
                </Grid>
                <Grid size={{xs: 12, sm: 12, md: 6}}>
                    <CardNotify
                        latestValue={ispu.value}
                    />
                </Grid>
            </Grid>

            {/* Second row */}
            <Grid
                container
                spacing={3}
                height={'auto'}
            >
                <Grid size={{xs: 12, sm: 12, md: 6}}>
                    <CardChart
                        heading="History & Forecast"
                        title="Consentration Overview"
                        subtitle="Last 1 Year"
                        series={[
                            { label: 'Actual', data: overview.actual },
                            { label: 'Predict', data: overview.prediction }
                        ]}
                    />
                </Grid>
                <Grid size={{xs: 12, sm: 12, md: 6}}>
                    <CardChart
                        heading="History & Forecast"
                        title="ISPU Overview"
                        subtitle="Last 1 Year"
                        series={[
                            { label: 'Actual', data: overview.actual },
                            { label: 'Predict', data: overview.prediction }
                        ]}
                    />
                </Grid>
            </Grid>

            {/* Third row */}
            <Grid
                container 
                spacing={3}
                height={'auto'}
            >
                <Grid size={{xs: 12, sm: 12, md: 6}}>
                    <CardChart
                        heading="History & Forecast"
                        title="ISPU Overview"
                        subtitle="Last 1 Year"
                        series={[
                            { label: 'Actual', data: overview.actual },
                            { label: 'Predict', data: overview.prediction }
                        ]}
                    />
                </Grid>

            </Grid>
        </Stack>
    );
}
    