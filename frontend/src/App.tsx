import { useState } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import Header from './components/Header/Header';
import AppContent from './components/AppContent/AppContent';
import './App.css';
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";


function App() {
    const [mode, setMode] = useState<'light' | 'dark'>('dark');

    const theme = createTheme({
        palette: {
        mode,
        ...(mode === 'light')
            ? {
                primary: {main: '#6F63FF'},
                secondary : {main: '#FFFFFF'},
                background : {default: '#ECECEC'},
                info : {main: '#FFFFFF'}
            }
            : {
                primary : {main: '#CFFCF4'},
                secondary : {main: '#212121'},
                background : {default: '#18181A'},
                info: {main: '#212121'}
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
            {/* Header */}
            <Header mode={mode} setMode={setMode}/>

            <div className="app-layout">
                {/* Left Navbar */}
                <Navbar/>

                {/* Main Content */}
                <div className="app-content">
                    <AppContent />
                </div>
            </div>
        </Router>
        </ThemeProvider>
    );
}

export default App;
