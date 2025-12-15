import Card from '@mui/material/Card';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import { alpha, useTheme } from '@mui/material/styles';
import { Box, Grid, Stack } from '@mui/material';
import ErrorOutlineRoundedIcon from '@mui/icons-material/ErrorOutlineRounded';
import Tooltip from '@mui/material/Tooltip';
import { ReactComponent as RobotIcon1 } from '../../../assets/robotlv1.svg';
import { ReactComponent as RobotIcon2 } from '../../../assets/robotlv2.svg';
import { ReactComponent as RobotIcon3 } from '../../../assets/robotlv3.svg';
import { ReactComponent as RobotIcon4 } from '../../../assets/robotlv4.svg';
import { ReactComponent as RobotIcon5 } from '../../../assets/robotlv5.svg';


function ISPUCard() {
    const theme = useTheme();
    const ispu = 301
    const consentration = 24.5
    const temperature = 40
    const humidity = 80
    const windSpeed = 50
    const windDirection = 120

    const getIspuCategory = (value: number) => {
        if (value <= 50) return 'Healthy'
        if (value <= 100) return 'Moderate';
        if (value <= 200) return 'Not Healthy';
        if (value <= 300) return 'Very Not Healthy';
        return 'Dangerous'
    }

    const getIspuColor = (value: number) => {
        if (value <= 50) return theme.palette.success.main;
        if (value <= 100) return 'blue';
        if (value <= 200) return theme.palette.warning.main;
        if (value <= 300) return theme.palette.error.main;
        return 'black'
    }

    const getRobotIcon = (value: number) => {
        if (value <= 50) return RobotIcon1;
        if (value <= 100) return RobotIcon2;
        if (value <= 200) return RobotIcon3;
        if (value <= 300) return RobotIcon4;
        return RobotIcon5
    }

    const ispuCategory = getIspuCategory(ispu);
    const ispuColor = getIspuColor(ispu)
    const RobotIcon = getRobotIcon(ispu);
    
    return (
        <Card sx={{
            height: '100%', 
            padding: '24px',
            borderRadius: '22px',
            background: theme.palette.primary.main,
            color: theme.palette.info.main,
        }}>
            <Stack direction='column' spacing={0.4}>

                {/* Pollutan Consentration */}
                <Stack direction={{xs: 'column', md:'row'}} spacing={2} sx={{
                    justifyContent: 'space-between',
                }}>
                    <CardHeader title='Consentration' subheader={`${consentration} µg/m³`} 
                    sx={{
                        padding: 0,
                    }}
                    titleTypographyProps={{
                        variant: 'subtitle1',
                        fontWeight: 'bold'
                    }}
                    subheaderTypographyProps={{
                        color: theme.palette.info.main,
                        marginTop: '-6px'
                    }}
                    />
                        <Chip label='ISPU' sx={{
                            background: theme.palette.info.main
                        }}/>
                </Stack>

                {/* ISPU Number */}
                <Stack spacing={1.5} direction='row' alignItems='center'>
                    <RobotIcon style={{
                        width: '120px',
                        height: '120px',
                        color: theme.palette.info.main
                    }}/>
                    <Grid size={{md: 12}} gap={0}>
                        <Typography variant='h3' sx={{
                            fontWeight: 'bold'
                        }}>
                            {ispu}
                        </Typography>
                        <Typography variant='h6' sx={{
                            marginTop: '-12px',
                            display: 'flex',
                            direction: 'row'
                        }}>
                            {ispuCategory}
                            <Box
                                sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                backgroundColor: ispuColor,
                                marginLeft: '2px',
                                marginTop: '4px',
                                }}
                            />
                        </Typography>
                    </Grid>
                </Stack>
                
                {/* Weather Now */}
                <Grid container rowSpacing={1} columnSpacing={{ xs: 1, sm: 2, md: 3 }} sx={{
                    padding: 0
                }}>
                    <Grid size={6} sx={{
                        padding: 0
                    }}>
                        <Stack spacing={0} direction='column' sx={{
                            justifyContent: 'center',
                        }}>
                            <Stack>

                            </Stack>
                            <Typography variant='subtitle2' sx={{
                                color: alpha(theme.palette.info.main, 0.8),
                                marginBottom: '-4px'
                            }}>Temperature
                                <Tooltip title='Air temperature at 2 meters above ground in Celcius.' arrow>
                                    <ErrorOutlineRoundedIcon
                                    sx={{
                                        fontSize: '1rem',
                                        marginLeft: '0.2rem',
                                        color: theme.palette.info.main,
                                        cursor: 'pointer',
                                    }}
                                    />
                                </Tooltip>
                            </Typography>
                            <Typography sx={{fontWeight:'bold'}}>{temperature}°C</Typography>
                        </Stack>
                    </Grid>
                    <Grid size={6}>
                        <Stack spacing={0} direction='column' sx={{
                            justifyContent: 'center',
                        }}>
                            <Typography variant='subtitle2' sx={{
                                color: alpha(theme.palette.info.main, 0.8),
                                marginBottom: '-4px'
                            }}>Humidity
                                <Tooltip title='Relative humidity at 2 meters above ground.' arrow>
                                    <ErrorOutlineRoundedIcon
                                    sx={{
                                        fontSize: '1rem',
                                        marginLeft: '0.2rem',
                                        color: theme.palette.info.main,
                                        cursor: 'pointer',
                                    }}
                                    />
                                </Tooltip>
                            </Typography>
                            <Typography sx={{fontWeight:'bold'}}>{humidity}%</Typography>
                        </Stack>
                    </Grid>
                    <Grid size={6}>
                        <Stack spacing={0} direction='column' sx={{
                            justifyContent: 'center',
                        }}>
                            <Typography variant='subtitle2' sx={{
                                color: alpha(theme.palette.info.main, 0.8),
                                marginBottom: '-4px'
                            }}>Wind Speed
                                <Tooltip title='Wind speed at 10 meters above ground. Wind speed on 10 meters is the standard level.' arrow>
                                    <ErrorOutlineRoundedIcon
                                    sx={{
                                        fontSize: '1rem',
                                        marginLeft: '0.2rem',
                                        color: theme.palette.info.main,
                                        cursor: 'pointer',
                                    }}
                                    />
                                </Tooltip>
                            </Typography>
                            <Typography sx={{fontWeight:'bold'}}>{windSpeed} km/h</Typography>
                        </Stack>
                    </Grid>
                    <Grid size={6}>
                        <Stack spacing={0} direction='column' sx={{
                            justifyContent: 'center',
                        }}>
                            <Typography variant='subtitle2' sx={{
                                color: alpha(theme.palette.info.main, 0.8),
                                marginBottom: '-4px'
                            }}>Wind Direction
                                <Tooltip title='Wind direction at 10 meters above ground.' arrow>
                                    <ErrorOutlineRoundedIcon
                                    sx={{
                                        fontSize: '1rem',
                                        marginLeft: '0.2rem',
                                        color: theme.palette.info.main,
                                        cursor: 'pointer',
                                    }}
                                    />
                                </Tooltip>
                            </Typography>
                            <Typography sx={{fontWeight:'bold'}}>{windDirection}°</Typography>
                        </Stack>
                    </Grid>
                    </Grid>
            </Stack>
        </Card>
    )
}

export default ISPUCard;