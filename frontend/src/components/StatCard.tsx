import { Card, CardContent, Typography, Box } from '@mui/material';


export default function StatCard({ title, value, height = 140 }: any) {
    return (
        <Card sx={{ height }}>
            <CardContent>
                <Typography variant="subtitle2">{title}</Typography>
                    {value && (
                        <Typography variant="h4" fontWeight="bold">
                            {value}
                        </Typography>
                    )}
                <Box sx={{ mt: 2, height: 60, bgcolor: '#eee', borderRadius: 2 }} />
            </CardContent>
        </Card>
    );
}