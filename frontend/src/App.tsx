import { Box } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import Stack from '@mui/material/Stack';
import { BrowserRouter as Router } from 'react-router-dom';
import { ThemeProvider, createTheme } from "@mui/material/styles";
import AppNavbar from './components/AppNavbar';
import Header from './components/Header';
import MainGrid from './components/MainGrid';
import SideMenu from './components/SideMenu';
import { useState } from 'react';
import PollutanSelector from './components/PollutanSelector';

export default function App() {
    const [mode, setMode] = useState<'light' | 'dark'>('dark');
    const [pollutant, setPollutant] = useState<string>('PM2.5');

    const theme = createTheme({
        palette: {
        mode,
        ...(mode === 'light')
            ? {
                primary: {main: '#6F63FF'},
                secondary : {main: '#EDEEF4'},
                // background : {default: '#ECECEC'},
                // info : {main: '#EDEEF4'}
            }
            : {
                primary : {main: '#CFFCF4'},
                secondary : {main: '#212121'},
                background : {default: '#18181A'},
                // info: {main: '#212121'}
            }
        },
        typography: {
        fontFamily: 'Poppins, Arial, sans-serif'
        }
    });

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Router>
                <Box sx={{ display: 'flex' }}>
                    <SideMenu />
                    <AppNavbar />
                    
                    {/* Main content */}
                    <Box
                        component="main"
                        sx={{
                            flexGrow: 1,
                            backgroundColor: theme.palette.background.paper
                        }}
                    >
                        <Stack
                            spacing={2}
                            sx={{
                                alignItems: 'center',
                                mx: 3,
                                pb: 5,
                                mt: { xs: 8, md: 0 },
                            }}
                        >
                            <Header mode={mode} setMode={setMode} pollutant={pollutant} setPollutant={setPollutant}/>
                            <svg
                                style={{
                                position: 'absolute',
                                width: 0,
                                height: 0,
                                overflow: 'hidden',
                                }}
                            >
                                <defs>
                                <linearGradient id="gradientFill" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={theme.palette.primary.main} stopOpacity={0.6} />
                                    <stop offset="100%" stopColor={theme.palette.primary.main} stopOpacity={0} />
                                </linearGradient>
                                </defs>
                            </svg>

                            <MainGrid pollutant={pollutant}/>
                        </Stack>
                    </Box>
                </Box>
            </Router>
        </ThemeProvider>
    );
}