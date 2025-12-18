import Stack from '@mui/material/Stack';
import NavbarBreadcrumbs from './NavbarBreadcrumbs';
import ThemeToggle from './ThemeToggle';
import PollutanSelector from './PollutanSelector';

type HeaderProps = {
    mode: "light" | "dark";
    setMode: React.Dispatch<React.SetStateAction<"light" | "dark">>;
    pollutant: string;
    setPollutant: React.Dispatch<React.SetStateAction<string>>;

};

export default function Header({ mode, setMode, pollutant, setPollutant  }: HeaderProps) {
    
    return (
        <Stack
            direction="row"
            spacing={2}
            justifyContent={'space-between'}
            sx={{
                display: { xs: 'none', md: 'flex' },
                width: '100%',
                alignItems: { xs: 'flex-start', md: 'center' },
                justifyContent: 'space-between',
                maxWidth: { sm: '100%', md: '1700px' },
                pt: 1.5,
            }}
        >
            <NavbarBreadcrumbs />
            <PollutanSelector selected={pollutant} setSelected={setPollutant} />
            <ThemeToggle mode={mode} setMode={setMode} />
        </Stack>
    );
}
