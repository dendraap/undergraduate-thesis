import { Grid } from "@mui/material";
import ISPUCard from './cards/ISPUCard'
import DailyConsentrationChart from "./cards/DailyConsentrationChart";

function AppContent() {
    return (
        <Grid container spacing={2}>
            <Grid container sx={{height: 'auto'}}>
                <Grid container size={{xs:12, md:3.5 }}>
                    <Grid size={{xs: 12, md: 12}}>
                        <ISPUCard />
                    </Grid>
                </Grid>
                <Grid size={{xs:12, md: 8.5}}>
                    <DailyConsentrationChart />
                </Grid>
            </Grid>
            <Grid container size={{xs:12, md: 4}}>
                <Grid size={{xs: 12, md: 12}}>
                    <ISPUCard />
                </Grid>
                <Grid size={{xs: 12, md: 12}}>
                    <ISPUCard />
                </Grid>
            </Grid>
            <Grid container size={{xs:12, md: 4}}>
                <Grid size={{xs: 12, md: 12}}>
                    <ISPUCard />
                </Grid>
                <Grid size={{xs: 12, md: 12}}>
                    <ISPUCard />
                </Grid>
            </Grid>
            <Grid container size={{xs:12, md: 4}}>
                <Grid size={{xs: 12, md: 12}}>
                    <ISPUCard />
                </Grid>
            </Grid>
        </Grid>
    );
}

export default AppContent;