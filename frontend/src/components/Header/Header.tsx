import { useState } from "react";
import ThemeToggle from "../ThemeToggle";
import { useTheme } from "@mui/material/styles";
import Paper from "@mui/material/Paper";
import MenuList from "@mui/material/MenuList";
import MenuItem from "@mui/material/MenuItem";
import ListItemText from "@mui/material/ListItemText";
import { alpha } from "@mui/material/styles";

const pollutants = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"];

type HeaderProps = {
    mode: "light" | "dark";
    setMode: React.Dispatch<React.SetStateAction<"light" | "dark">>;
};

function Header({ mode, setMode }: HeaderProps) {
    const [selected, setSelected] = useState(pollutants[0]);
    const theme = useTheme();

    return (
        <header style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            background: theme.palette.background.default,
            padding: "12px 24px",
            margin: 0,
            position: 'fixed',
            top: 0,
            width: '100%'
        }}>
        {/* Dashboard Title */}
        <div style={{
            borderRadius: 100,
            background: theme.palette.primary.main,
            padding: "12px 24px",
        }}>
            <b style={{ color: theme.palette.info.main }}>Air Quality Dashboard</b>
        </div>

        {/* Pollutants Menu */}
        <Paper sx={{
            borderRadius: "100px",
            background: theme.palette.secondary.main,
            boxShadow: "none",
        }}>
            <MenuList sx={{ display: "flex", flexDirection: "row", padding: 0 }}>
            {pollutants.map((p) => (
                <MenuItem
                key={p}
                selected={selected === p}
                onClick={() => setSelected(p)}
                sx={{
                    borderRadius: "100px",
                    padding: "12px 24px",
                    transition: "0.2s",
                    "&.Mui-selected": {
                        backgroundColor: theme.palette.primary.main,
                        color: theme.palette.info.main,
                    },
                    "&.Mui-selected:hover": {
                        backgroundColor: alpha(theme.palette.primary.main, 0.7),
                        color: theme.palette.info.main,
                    },
                    "&:not(.Mui-selected):hover": {
                        backgroundColor: "transparent",
                    },
                }}>
                    <ListItemText primary={p} />
                </MenuItem>
            ))}
            </MenuList>
        </Paper>

        {/* Toggle Theme */}
        <ThemeToggle mode={mode} setMode={setMode} />
        </header>
    );
}

export default Header;