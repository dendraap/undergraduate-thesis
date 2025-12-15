import { Link } from "react-router-dom";
import { useTheme } from "@mui/material/styles";
import { useState } from "react";
import Paper from '@mui/material/Paper';
import MenuList from '@mui/material/MenuList';
import MenuItem from '@mui/material/MenuItem';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import ContentCut from '@mui/icons-material/ContentCut';
import ContentCopy from '@mui/icons-material/ContentCopy';
import ContentPaste from '@mui/icons-material/ContentPaste';
import { alpha } from "@mui/material/styles";


function Navbar() {
    const theme = useTheme();
    const [active, setActive] = useState('tft');

    return (
        // </nav>
        <Paper sx={{ 
            width: 300, 
            maxWidth: '100%', 
            borderRadius: '24px',
            background: theme.palette.info.main,
            color: theme.palette.text.primary,
            height: "calc(97vh - 88px)",
            top: '88px',
            position: "fixed",
            padding: '24px',
            boxSizing: "border-box",
            boxShadow: 'none'
            }}>
            <MenuList sx={{
                
            }}>
                <MenuItem 
                    component={Link} 
                    to='/tft'
                    selected={active === 'tft'}
                    onClick={() => setActive('tft')}
                    sx={{
                        borderRadius: '24px',
                        padding: '12px 24px',
                        marginBottom: '12px',
                        '&.Mui-selected': {
                            backgroundColor: theme.palette.primary.main,
                            color: theme.palette.getContrastText(theme.palette.primary.main),
                            '& svg': {
                                color: theme.palette.info.main
                            }
                        },
                        '&.Mui-selected:hover': {
                            backgroundColor: alpha(theme.palette.primary.main, 0.7)
                        }
                    }}
                >
                    <ListItemIcon>
                        <ContentCopy fontSize="small"/>
                    </ListItemIcon>
                    <ListItemText>TFT Model</ListItemText>
                </MenuItem>
                <MenuItem
                    component={Link} 
                    to='/nbeats'
                    selected={active === 'nbeats'}
                    onClick={() => setActive('nbeats')}
                    sx={{
                        borderRadius: '24px',
                        padding: '12px 24px',
                        marginBottom: '12px',
                        '&.Mui-selected': {
                            backgroundColor: theme.palette.primary.main,
                            color: theme.palette.getContrastText(theme.palette.primary.main),
                            '& svg': {
                                color: theme.palette.info.main
                            }
                        },
                        '&.Mui-selected:hover': {
                            backgroundColor: alpha(theme.palette.primary.main, 0.7)
                        }
                    }}
                >
                    <ListItemIcon>
                        <ContentCopy fontSize="small"/>
                    </ListItemIcon>
                    <ListItemText>N-BEATS Model</ListItemText>
                </MenuItem>
                <MenuItem
                    component={Link} 
                    to='/about'
                    selected={active === 'about'}
                    onClick={() => setActive('about')}
                    sx={{
                        borderRadius: '24px',
                        padding: '12px 24px',
                        marginBottom: '12px',
                        '&.Mui-selected': {
                            backgroundColor: theme.palette.primary.main,
                            color: theme.palette.getContrastText(theme.palette.primary.main),
                            '& svg': {
                                color: theme.palette.info.main
                            }
                        },
                        '&.Mui-selected:hover': {
                            backgroundColor: alpha(theme.palette.primary.main, 0.7)
                        }
                    }}
                >
                    <ListItemIcon>
                        <ContentPaste fontSize="small" />
                    </ListItemIcon>
                    <ListItemText>About</ListItemText>
                </MenuItem>
                <MenuItem 
                    component='a' 
                    href='https://github.com/dendraap/thesis'
                    target='_blank'
                    sx={{
                        borderRadius: '24px',
                        padding: '12px 24px',
                        marginBottom: '12px',
                        '&.Mui-selected': {
                            backgroundColor: theme.palette.primary.main,
                            color: theme.palette.getContrastText(theme.palette.primary.main),
                            '& svg': {
                                color: theme.palette.info.main
                            }
                        },
                        '&.Mui-selected:hover': {
                            backgroundColor: alpha(theme.palette.primary.main, 0.7)
                        }
                    }}
                >
                    <ListItemIcon>
                        <ContentPaste fontSize="small" />
                    </ListItemIcon>
                    <ListItemText>GitHub</ListItemText>
                </MenuItem>
            </MenuList>
        </Paper>
    );
}

export default Navbar;