import { alpha, useTheme } from '@mui/material/styles'
import { Stack, Card, CardContent, Typography, Chip } from '@mui/material'
import { SparkLineChart } from '@mui/x-charts/SparkLineChart'
import { areaElementClasses, lineElementClasses } from '@mui/x-charts/LineChart'
import CircleIcon from '@mui/icons-material/Circle'


interface CardMiniProps {
    title: string
    latestValue: number | null
    unit?: string
    deltaPercent?: number | null
    series?: number[]
    window?: string
    category?: string
    showDot?: boolean
}

export default function CardMini({
    title,
    latestValue,
    deltaPercent = null,
    series = [],
    window,
    unit,
    showDot = false,
    category,
}: CardMiniProps) {
    const theme = useTheme();
    const getIspuColor = (category?: string) => {
        if (!category) return theme.palette.text.secondary;

        switch (category.toLowerCase()) {
        case 'sehat':
            return theme.palette.success.main;
        case 'sedang':
            return theme.palette.info.main;
        case 'tidak sehat':
            return theme.palette.warning.main;
        case 'sangat tidak sehat':
            return theme.palette.error.main;
        case 'berbahaya':
            return theme.palette.error.dark;
        default:
            return theme.palette.text.primary;
        }
    };

    const getDeltaColor = (percent: number | null) => {
        if (percent === null) return theme.palette.text.secondary;
        if (percent < 0) return theme.palette.success.main;
        if (percent <= 25) return theme.palette.info.main;
        if (percent <= 50) return theme.palette.warning.main;
        return theme.palette.error.main;
    };

    const ispuColor = getIspuColor(category);
    const deltaColor = getDeltaColor(deltaPercent);

    return (
        <Card
            variant="outlined"
            sx={{
                flexShrink: 0,
                background: theme.palette.background.paper,
                height: '100%',
                borderRadius: '18px'
            }}
        >
            <CardContent
                sx={{
                    p: 2, pb: 2,
                    '&.MuiCardContent-root': {
                        paddingBottom: '16px',
                    },
                }}
            >
                <Typography variant="body1">{title}</Typography>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                    {showDot ? (
                        <Stack direction="row" spacing={1} alignItems="center">
                            <CircleIcon fontSize="small" sx={{ color: ispuColor }} />
                            <Typography variant="h4">
                                {latestValue !== null ? latestValue : 'N/A'} {unit ?? ''}
                            </Typography>
                        </Stack>
                    ) : (
                        <Typography variant="h4">
                            {latestValue !== null ? latestValue : 'N/A'} {unit ?? ''}
                        </Typography>
                    )}

                    <Chip
                        label={
                            deltaPercent !== null
                                ? `${deltaPercent > 0 ? '+' : ''}${deltaPercent.toFixed(1)}%`
                                : 'N/A'
                        }
                        sx={{
                            color: deltaColor,
                            backgroundColor: alpha(deltaColor, 0.2),
                            fontWeight: theme.typography.fontWeightMedium,
                            border: `1px solid ${deltaColor}`,
                        }}
                    />
                </Stack>
                <Typography variant="subtitle2" sx={{color: alpha(theme.palette.text.primary, 0.5)}}>Last {series.length} {window}s</Typography>

                {series.length > 0 ? (
                    <SparkLineChart
                        height={50}
                        area
                        showTooltip
                        showHighlight
                        axisHighlight={{ x: 'line' }}
                        color={theme.palette.primary.main}
                        // data={[3, -10, -2, 5, 7, -2, 4, 6]}
                        data={series}
                        sx={{
                            [`& .${areaElementClasses.root}`]: {
                                fill: 'url(#gradientFill)',
                            },
                            [`& .${lineElementClasses.root}`]: {
                                strokeWidth: 2,
                            },
                        }}
                    />
                ) : (
                    <Typography variant="caption" color="text.secondary">
                        No data available
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
}
