import { useState } from 'react';
import { Box, CssBaseline, Stack } from '@mui/material';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from "@mui/material/styles";
import AppNavbar from './components/AppNavbar';
import SideMenu from './components/SideMenu';
import TFTPage from './pages/TFTPage';
import NBEATSPage from './pages/NBEATSPage';
import Header from './components/Header';


export default function App() {
    const [mode, setMode] = useState<'light' | 'dark'>('dark');
    const [pollutantKey, setPollutantKey] = useState<string>('pm25');

    const theme = createTheme({
        palette: {
            mode,
            ...(mode === 'light')
                ? {
                    primary: {main: '#6F63FF'},
                    secondary : {main: '#EDEEF4'},
                }
                : {
                    primary : {main: '#CFFCF4'},
                    secondary : {main: '#212121'},
                    background : {default: '#18181A'},
                }
            },
        typography: { fontFamily: 'Poppins, Arial, sans-serif' }
    });

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <BrowserRouter>
                <Box sx={{ display: 'flex' }}>
                    <SideMenu />
                    <AppNavbar />

                    <Box component="main" sx={{ flexGrow: 1 }}>
                        <Box
                            sx={{
                                position: "sticky",
                                top: 0,
                                zIndex: theme.zIndex.appBar,
                                backgroundColor: theme.palette.background.default,
                                mx: 3
                            }}
                        >
                            <Header
                                mode={mode}
                                setMode={setMode}
                                pollutantKey={pollutantKey}
                                setPollutantKey={setPollutantKey}
                            />
                        </Box>
                        <Stack spacing={2} sx={{ mx: 3, mt: { xs: 8, md: 0 } }}>
                            {/* <Header mode={mode} setMode={setMode} pollutantKey={pollutantKey} setPollutantKey={setPollutantKey}/> */}
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
                            <Routes>
                                <Route path="/tft" element={<TFTPage pollutantKey={pollutantKey}/>} />
                                <Route path="/nbeats" element={<NBEATSPage />} />
                            </Routes>
                        </Stack>
                    </Box>
                </Box>
            </BrowserRouter>
        </ThemeProvider>
    );
}
