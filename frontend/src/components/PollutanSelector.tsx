import { ToggleButton, ToggleButtonGroup, useTheme } from '@mui/material';
import { POLLUTANTS } from "../constants/pollutants";

interface PollutanSelectorProps {
    selectedKey: string;
    setSelectedKey: (key: string) => void;
}

export default function PollutanSelector({ selectedKey, setSelectedKey }: PollutanSelectorProps) {
    const theme = useTheme();

    return (
        <ToggleButtonGroup
            value={selectedKey}
            exclusive
            onChange={(_, key) => key && setSelectedKey(key)}
            sx={{
                mb: 2,
                gap: 0.5,
                background: theme.palette.secondary.main,
                borderRadius: '999px',
                '& .MuiToggleButton-root': {
                    border: 'none',
                    borderRadius: '999px', 
                    px: 2.4,
                    py: 1,
                    background: theme.palette.secondary.main
                },
                '& .MuiToggleButton-root.Mui-selected': {
                    backgroundColor: theme.palette.primary.main,
                    color: theme.palette.getContrastText(theme.palette.text.primary),
                },
            }}
        >
            {POLLUTANTS.map(p => (
                <ToggleButton
                    key={p.key}
                    value={p.key}
                    sx={{ borderRadius: '999px', px: 2.4 }}
                >
                    {p.label}
                </ToggleButton>
            ))}
        </ToggleButtonGroup>
    );
}
