import Card from '@mui/material/Card';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import { alpha, useTheme } from '@mui/material/styles';
import { Box, Grid, Stack } from '@mui/material';
import ErrorOutlineRoundedIcon from '@mui/icons-material/ErrorOutlineRounded';
import Tooltip from '@mui/material/Tooltip';
import { LineChart, lineElementClasses } from '@mui/x-charts/LineChart';

function DailyConsentrationChart() {
    const theme = useTheme();
    const margin =  {left: 0, right: 0};
    const uData = [53.5, 45, 55, 30, 60, 54, 56, 49, 49, 40, 59, 58, 30, 49, 50, 38, 85, 48];
    const xLabels = [
        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00',
        '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00',
        '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00',
        '21:00', '22:00', '23:00', '24:00'

    ];
    return (
        <Card sx={{
            height: '314px',
            padding: '24px',
            borderRadius: '22px',
            boxShadow: 'none',
            background: theme.palette.info.main,
            color: theme.palette.info.main,
        }}>
            <Stack direction='row' justifyContent='space-between'>
                <Typography variant='h6' sx={{
                    color: theme.palette.getContrastText(theme.palette.info.main),
                    fontWeight: 'bold'
                }}>
                    PM 2.5 Consentration
                </Typography>
                <Chip label='Daily' sx={{
                    background: theme.palette.primary.main,
                    color: theme.palette.info.main
                }}/>
            </Stack>

            <Box sx={{ width: '100%', height: 250 }}>
                <LineChart
                    series={[{ 
                        data: uData, area: true, showMark: false, color: theme.palette.primary.main }
                    ]}
                    xAxis={[{ scaleType: 'point', data: xLabels }]}
                    sx={{
                        [`& .MuiAreaElement-root`]: {
                            fill: 'url(#gradientArea)',
                        },
                    }}
                    margin={margin}
                />
                <svg width='0' height='0'>
                    <defs>     
                        <linearGradient id='gradientArea' x1='0' y1='0' x2='0' y2='1'>
                            <stop offset='30%' stopColor={alpha(theme.palette.primary.dark, 1)} />
                            <stop offset='100%' stopColor={alpha(theme.palette.primary.dark, 0.2)} />
                        </linearGradient>
                    </defs>
                </svg>
            </Box>
                        
        </Card>
    )
}

export default DailyConsentrationChart