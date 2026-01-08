import { Box, Card, CardContent, Typography, useTheme } from "@mui/material";
import { LineChart, lineElementClasses } from "@mui/x-charts";


interface RawPoint {
    t: string
    v: number
}

interface ChartSeries {
    label: string
    data: RawPoint[]
}

interface CardChartProps {
    heading: string
    title: string
    subtitle?: string
    series: ChartSeries[]
}

export default function CardChart({ heading, title, subtitle, series }: CardChartProps) {
    const theme = useTheme();

    const xAxisData =
        series[0]?.data
            ?.map(p => p.t)
            .filter(Boolean) ?? [];

    const chartSeries = series.map(s => ({
        label: s.label,
        data: s.data
            .map(p => p.v)
            .filter(v => v !== null && v !== undefined),
        area: true,
        showMark: false,
        stack: "total",
    }));

    const isValid =
        xAxisData.length > 0 &&
        chartSeries.every(s => s.data.length === xAxisData.length);

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
                    p: 2,
                    '&.MuiCardContent-root': {
                        paddingBottom: '16px',
                    },
                }}
            >
                <Typography variant="body1">{heading}</Typography>
                <Typography variant="h4">{title}</Typography>

                {subtitle && (
                    <Typography variant="subtitle2" color={theme.palette.text.secondary}>
                        {subtitle}
                    </Typography>
                )}

                {isValid ? (
                    <Box>
                        <LineChart
                            series={chartSeries}
                            xAxis={[{ scaleType: 'point', data: xAxisData }]}
                            yAxis={[{ width: 50 }]}
                            height={250}
                            sx={{
                                [`& .${lineElementClasses.root}`]: {
                                display: 'none',
                                },
                            }}
                        />
                    </Box>
                ) : (
                <Typography variant="caption" color="text.secondary">
                    No data available
                </Typography>
                )}
            </CardContent>
        </Card>
    );
}
