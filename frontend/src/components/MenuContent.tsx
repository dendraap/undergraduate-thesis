import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import { Link, useLocation  } from "react-router-dom";
import Stack from '@mui/material/Stack';
import HomeRoundedIcon from '@mui/icons-material/HomeRounded';
import AnalyticsRoundedIcon from '@mui/icons-material/AnalyticsRounded';
import InfoRoundedIcon from '@mui/icons-material/InfoRounded';  
import HelpRoundedIcon from '@mui/icons-material/HelpRounded';
import { useTheme } from '@mui/material/styles';

const mainListItems = [
    { text: 'TFT Model', link: '/tft', icon: <HomeRoundedIcon /> },
    { text: 'N-BEATS Model', link: '/nbeats', icon: <AnalyticsRoundedIcon /> },
    { text: 'About', link: '/about', icon: <InfoRoundedIcon /> },
];

const secondaryListItems = [
    { text: 'GitHub', link: 'https://github.com/dendraap/thesis',icon: <HelpRoundedIcon /> },
];

export default function MenuContent() {
    const theme = useTheme()
    const location = useLocation()

    return (
        <Stack sx={{ flexGrow: 1, p: 1, justifyContent: 'space-between' }}>
            <List dense>
                {mainListItems.map((item, index) => (
                    <ListItem 
                        key={index} 
                        disablePadding
                        component={Link}
                        to={item.link} 
                        sx={{ 
                                display: 'block',
                                textDecoration: 'none',
                                color: 'inherit'
                            }}
                    >
                        <ListItemButton 
                            selected={location.pathname === item.link}
                            sx={{
                                '&.Mui-selected': {
                                    backgroundColor: theme.palette.primary.main,
                                    color: theme.palette.getContrastText(theme.palette.text.primary),
                                '& .MuiListItemIcon-root': {
                                    color: theme.palette.getContrastText(theme.palette.text.primary),
                                },
                                },
                                '&.Mui-selected:hover': {
                                    backgroundColor: theme.palette.primary.dark,
                                },
                            }}
                        >
                            <ListItemIcon>{item.icon}</ListItemIcon>
                            <ListItemText primary={item.text} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
            <List dense>
                <Divider />
                {secondaryListItems.map((item, index) => (
                    <ListItem 
                        key={index} 
                        disablePadding
                        component={Link}
                        to={item.link}
                        target='_blank'
                        sx={{ 
                            display: 'block',
                            textDecoration: 'none',
                            color: 'inherit'
                        }}
                    >
                        <ListItemButton selected={location.pathname === item.link}>
                            <ListItemIcon>{item.icon}</ListItemIcon>
                            <ListItemText primary={item.text} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Stack>
    );
}
