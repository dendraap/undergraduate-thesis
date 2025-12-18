import { useState } from 'react';
import { alpha, useTheme } from '@mui/material/styles';
import { Stack, Card, CardContent, Typography, Chip } from '@mui/material';
import { SparkLineChart } from '@mui/x-charts/SparkLineChart';
import { areaElementClasses, lineElementClasses } from '@mui/x-charts/LineChart';
import CircleIcon from '@mui/icons-material/Circle';
import { chartsAxisHighlightClasses } from '@mui/x-charts/ChartsAxisHighlight';

interface DataPoint {
    timestamp: string;
    value: number;
}

interface CardMiniProps {
    title: string;
    last: number;
    percentage: number;
    data?: DataPoint[];
    dot?: boolean;
    unit?: string;
    hourly?: boolean;
    daily?: boolean;
}

export default function CardMini({
    title, last, data, percentage, dot, unit, hourly, daily
}: CardMiniProps) {
    const theme = useTheme();
    const values = data?.map((point) => point.value) ?? [];
    const labels = data?.map((point) => point.timestamp) ?? [];
    const lastValue = values.length > 0 ? values[values.length - 1] : null;
    const prevValue = values.length > 1 ? values[values.length - 2] : null;
    const mae = 0.2

    const percentageChange =
        lastValue !== null && prevValue !== null
            ? ((lastValue - prevValue) / prevValue) * 100
            : null;

    const getIspuColor = (value: number | null) => {
        if (value === null) return theme.palette.text.secondary;
        if (value <= 50) return theme.palette.success.main;
        if (value <= 100) return theme.palette.info.main;
        if (value <= 200) return theme.palette.warning.main;
        if (value <= 300) return theme.palette.error.main;
        return theme.palette.text.primary
    };

    const getPercentageColor = (percent: number | null) => {
        if (percent === null) return theme.palette.text.secondary;
        if (percent < 0) return theme.palette.success.main;
        if (percent <= 25) return theme.palette.info.main;
        if (percent <= 50) return theme.palette.warning.main;
        return theme.palette.error.main
    };

    const getMaeColor = (mae: number) => {
        if (mae <= 0.05) return theme.palette.success.main;
        if (mae <= 0.1) return theme.palette.primary.main;
        if (mae <= 0.2) return theme.palette.warning.main;
        return theme.palette.error.main;
    };

    const ispuColor = getIspuColor(lastValue);
    const maeColor = getMaeColor(mae)
    const percentageColor = getPercentageColor(percentageChange )

    return (
        <Card 
            variant="outlined" 
            sx={{ 
                flexShrink: 0,
                background: theme.palette.background.paper,
                height: '100%'
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
                <Typography variant='body1'>
                    {title}
                </Typography>
                <Stack direction={'row'} justifyContent={'space-between'}>
                    {dot === true ? (
                        <Stack direction={'row'} spacing={1} alignItems={'center'}>
                            <CircleIcon fontSize='small' sx={{color: ispuColor}}/>
                            <Typography variant='h4'>
                                {lastValue !== null ? lastValue : 'N/A'} {unit ?? ''}
                            </Typography>
                        </Stack>
                    ) : (
                        <Typography variant='h4'>
                            {lastValue !== null ? lastValue : 'N/A'} {unit ?? ''}
                        </Typography>
                    )}
                    <Chip 
                        label={
                            percentageChange !== null
                            ? `${percentageChange > 0 ? '+' : ''}${percentageChange.toFixed(1)}%`
                            : 'N/A'
                        }
                        sx={{
                            color: percentageColor,
                            backgroundColor: alpha(percentageColor, 0.3),
                            fontWeight: theme.typography.fontWeightMedium,
                            border: `1px solid ${percentageColor}`,
                        }}
                    />
                </Stack>
                <Typography variant='subtitle2' color={theme.palette.text.secondary}>
                    Last {last} days
                </Typography>
                {values.length > 0 ? (
                    <SparkLineChart
                        height={50}
                        area
                        showTooltip
                        showHighlight
                        axisHighlight={{ x: 'line' }}
                        color={theme.palette.primary.main}
                        data={values}
                        xAxis={{
                            id: 'day-axis',
                            data: labels,
                        }}
                        sx={{
                            [`& .${areaElementClasses.root}`]: {
                            fill: 'url(#gradientFill)',
                            },
                            [`& .${lineElementClasses.root}`]: { strokeWidth: 2 },
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
