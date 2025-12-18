import { Box, Card, CardContent, Typography, useTheme } from "@mui/material";
import { LineChart, lineElementClasses  } from "@mui/x-charts";

interface DataPoint {
    timestamp: string;
    value: number;
}

interface CardChartProps {
    actualData?: DataPoint[];
    predictData?: DataPoint[];
    dataContext: string;
}

export default function CardChart ({actualData, predictData, dataContext}: CardChartProps) {
    const theme = useTheme();
    const valuesActual  = actualData?.map((point) => point.value) ?? [];
    const valuesPredict = predictData?.map((point) => point.value) ?? [];
    const labelsActual  = actualData?.map((point) => point.timestamp) ?? [];
    const labelsPredict = predictData?.map((point) => point.timestamp) ?? [];
    
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
                <Typography variant="body1">
                    Overview
                </Typography>
                <Typography variant="h4">
                    {dataContext}
                </Typography>
                <Typography variant='subtitle2' color={theme.palette.text.secondary}>
                    Last 30 days
                </Typography>
                {valuesActual.length > 0 ? (
                    <Box>
                        <LineChart 
                            series={[
                                { data: valuesActual, label: 'Actual', area: true, stack: 'Total', showMark: false },
                                { data: valuesPredict, label: 'Predict', area: true, stack: 'Total', showMark: false}
                            
                            ]}
                            xAxis={[{ scaleType: 'point', data: labelsPredict }]}
                            yAxis={[{ width: 50 }]}
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
                    )
            
                }
            </CardContent>

        </Card>
    )
}