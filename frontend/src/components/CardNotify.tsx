import { alpha, Box, Card, CardContent, Stack, Typography, useTheme } from "@mui/material";
import PriorityHighRoundedIcon from '@mui/icons-material/PriorityHighRounded';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';

interface DataPoint {
    timestamp: string;
    value: number;
}

interface CardNotifyProps {
    data?: DataPoint[],
}

export default function CardNotify({data}: CardNotifyProps) {
    const theme = useTheme()
    const values = data?.map((point) => point.value) ?? [];
    const labels = data?.map((point) => point.timestamp) ?? [];
    const lastValue= values.length > 0 ? values[values.length - 1] : null;

    const getIspuColor = (value: number | null) => {
        if (value === null) return theme.palette.text.secondary;
        if (value <= 50) return theme.palette.success.main;
        if (value <= 100) return theme.palette.info.main;
        if (value <= 200) return theme.palette.warning.main;
        if (value <= 300) return theme.palette.error.main;
        return theme.palette.text.primary
    };

   const getIspuDesc = (value: number | null) => {
        if (value === null) {
            return 'Data not available.';
        }
        if (value <= 50) {
            return 'Air quality is good according to the Regulation of the Minister Environment and Forestry of the Republic Indonesia.';
        }
        if (value <= 100) {
            return 'PM2.5 ISPU is currently about twice the healthy standard, based on the Regulation of the MinisterEnvironment and Forestry of the Republic Indonesia.';
        }
        if (value <= 200) {
            return 'Air quality is unhealthy according to the Regulation of the Minister Environment and Forestry of the Republic Indonesia.';
        }
        if (value <= 300) {
            return 'Air quality is very unhealthy according to the Regulation of the Minister Environment and Forestry of the Republic Indonesia.';
        }
        return 'Air quality is hazardous according to the Regulation of the Minister Environment and Forestry of the Republic Indonesia.';
    };

    const ispuColor = getIspuColor(lastValue)
    const ispuDesc = getIspuDesc(lastValue)
    
    return (
        <Card
            variant="outlined" 
            sx={{ 
                flexShrink: 0,
                background: alpha(ispuColor, 0.1),
                height: '100%'
            }}
        >
            <CardContent
                sx={{ 
                    p: 2, pb: 2,
                    height: '100%',
                    alignContent: 'center',
                    '&.MuiCardContent-root': {
                        paddingBottom: '16px',
                    },
                }}    
            >
                <Stack direction={'row'} spacing={2} alignItems={'center'}>
                    <Box
                        alignContent={'center'}
                        justifyItems={'center'}
                        sx={{
                            background: alpha(ispuColor, 0.25),
                            borderRadius: '100%',
                            height: '70px',
                            width: '70px'
                        }}
                    >
                        <Box
                            display={'flex'}
                            alignItems={'center'}
                            justifyContent={'center'}
                            sx={{
                                background: ispuColor,
                                borderRadius: '100%',
                                height: '30px',
                                width: '30px'
                            }}
                        >
                            {lastValue === null ? (
                                <Typography variant="caption" color={theme.palette.getContrastText(ispuColor)}>
                                    N/A
                                </Typography>
                            ) : lastValue <= 50 ? (
                                    <CheckRoundedIcon
                                        sx={{
                                            color: theme.palette.getContrastText(ispuColor),
                                        }}
                                    />
                            ) : (
                                <PriorityHighRoundedIcon
                                    sx={{
                                        color: theme.palette.getContrastText(ispuColor),
                                    }}
                                />
                            )}
                        </Box>
                    </Box>
                    <Typography variant='body2' sx={{ flex: 1 }}>
                        {ispuDesc}
                    </Typography>
                </Stack>
            </CardContent>
        </Card>
    )
}