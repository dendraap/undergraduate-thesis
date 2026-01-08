import {Stack, Divider} from '@mui/material';
import NavbarBreadcrumbs from './NavbarBreadcrumbs';
import ThemeToggle from './ThemeToggle';
import PollutanSelector from './PollutanSelector';

type HeaderProps = {
    mode: 'light' | 'dark';
    setMode: React.Dispatch<React.SetStateAction<'light' | 'dark'>>;
    pollutantKey: string;
    setPollutantKey: (key: string) => void;
};

export default function Header({ mode, setMode, pollutantKey, setPollutantKey  }: HeaderProps) {
    return (
        <>
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
                    py: 1
                }}
            >
                <NavbarBreadcrumbs />
                <PollutanSelector selectedKey={pollutantKey} setSelectedKey={setPollutantKey} />
                <ThemeToggle mode={mode} setMode={setMode} />
            </Stack>
            <Divider />
        </>
    );
}
