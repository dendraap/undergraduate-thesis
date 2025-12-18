import { Grid } from '@mui/material';
import StatCard from './StatCard';

const items = ['Temperature', 'Humidity', 'Wind Speed', 'Rain'];

export default function WeatherCard() {
    return (
        <Grid container spacing={2}>
            {items.map((item) => (
                <Grid size={{xs: 12, sm: 6, md: 3}} key={item}>
                    <StatCard title={item} value="23" />
                </Grid>
            ))}
        </Grid>
    );
}