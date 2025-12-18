import { ToggleButton, ToggleButtonGroup, useTheme } from '@mui/material';

const pollutants = ['PM2.5', 'PM10', 'SO₂', 'NO₂', 'O₃', 'CO'];

interface PollutanSelectorProps {
    selected: string;
    setSelected: (value: string) => void;
}

export default function PollutanSelector({ selected, setSelected }: PollutanSelectorProps) {
    const theme = useTheme()
    const handleChange = (_: React.MouseEvent<HTMLElement>, newPollutant: string | null) => {
        if (newPollutant !== null) {
        setSelected(newPollutant);
        }
    };

    return (
        <ToggleButtonGroup
            value={selected}
            exclusive
            onChange={handleChange}
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
            {pollutants.map((p) => (
                <ToggleButton 
                    key={p} 
                    value={p} 
                    sx={{
                        background: theme.palette.background.paper
                    }}
                >
                    {p}
                </ToggleButton>
            ))}
        </ToggleButtonGroup>
    );
}