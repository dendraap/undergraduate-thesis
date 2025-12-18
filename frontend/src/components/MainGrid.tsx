import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material/styles';
import CardMini from './CardMini';
import CardNotify from './CardNotify';
import CardChart from './CardChart';

interface MainGridProps {
    pollutant: string;
}

export default function MainGrid({ pollutant }: MainGridProps) {
    const theme = useTheme()

    // dummy data
    const datasets: Record<string, { timestamp: string; value: number }[]> = {
        'PM2.5': Array.from({ length: 72 }, (_, i) => {
            const d = new Date(Date.now() - i * 60 * 60 * 1000);
                return { timestamp: d.toISOString(), value: Math.floor(20 + Math.random() * 400) };
            }).reverse(),
        'PM10': Array.from({ length: 72 }, (_, i) => {
            const d = new Date(Date.now() - i * 60 * 60 * 1000);
            return { timestamp: d.toISOString(), value: Math.floor(30 + Math.random() * 100) };
            }).reverse(),
        'SO₂': [
            { timestamp: '2025-12-10', value: 12 },
            { timestamp: '2025-12-11', value: 15 },
            { timestamp: '2025-12-12', value: 10 },
        ],
        'NO₂': [
            { timestamp: '2025-12-10', value: 30 },
            { timestamp: '2025-12-11', value: 28 },
            { timestamp: '2025-12-12', value: 25 },
        ],
        'O₃': [
            { timestamp: '2025-12-10', value: 50 },
            { timestamp: '2025-12-11', value: 45 },
            { timestamp: '2025-12-12', value: 40 },
        ],
        'CO': [
            { timestamp: '2025-12-10', value: 5 },
            { timestamp: '2025-12-11', value: 6 },
            { timestamp: '2025-12-12', value: 4 },
        ],
    };

    const data = datasets[pollutant] ?? [];

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
                        title='ISPU' 
                        last={7} 
                        data={data}
                        percentage={34}
                        dot={true}
                        daily={true}
                    />
                </Grid>
                <Grid size={{xs: 12, sm: 6, md: 3}}>
                    <CardMini 
                        title='Consentration' 
                        last={3}
                        data={data} 
                        percentage={34}
                    />
                </Grid>
                <Grid size={{xs: 12, sm: 12, md: 6}}>
                    <CardNotify 
                        data={data}
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
                        actualData={data}
                        predictData={data}
                        dataContext={pollutant}
                    />
                </Grid>
                <Grid size={{xs: 12, sm: 12, md: 6}}>

                </Grid>
            </Grid>
        </Stack>
    );
}
