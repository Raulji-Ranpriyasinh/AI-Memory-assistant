import { useTranslation } from 'react-i18next';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';

interface GlucoseChartProps {
  data: { timestamp: string; glucoseMgDl: number }[];
}

export default function GlucoseChart({ data }: GlucoseChartProps) {
  const { t, i18n } = useTranslation('cgm');

  // Use Intl.DateTimeFormat for locale-aware time formatting
  const timeFormatter = new Intl.DateTimeFormat(i18n.language, {
    hour: 'numeric',
    minute: '2-digit',
    hour12: !['de', 'fr', 'ru', 'nl', 'he', 'ar', 'tr', 'ja', 'zh'].includes(i18n.language),
  });

  const chartData = data
    .map((d) => ({
      ...d,
      time: timeFormatter.format(new Date(d.timestamp)),
    }))
    .reverse();

  // Locale-aware glucose value formatting
  const fmtValue = (v: number) =>
    new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 0 }).format(v);

  const getStrokeColor = (value: number) => {
    if (value < 70 || value > 180) return '#DC2626';
    if (value > 140) return '#EAB308';
    return '#16A34A';
  };

  return (
    <div
      role="img"
      aria-label={t('chart.ariaLabel', { period: '24h' })}
      style={{ direction: 'ltr' }}
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="time" stroke="#64748b" fontSize={11} tick={{ dy: 4 }} />
          <YAxis
            stroke="#64748b"
            fontSize={11}
            domain={[40, 250]}
            tickFormatter={fmtValue}
            label={{
              value: t('chart.yAxisLabel', { unit: 'mg/dL' }),
              angle: -90,
              position: 'insideLeft',
              fill: '#94a3b8',
              fontSize: 10,
              dx: -4,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e2e8f0',
              borderRadius: '10px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
              fontSize: '13px',
            }}
            formatter={(value: any) => [
              `${fmtValue(value)} mg/dL`,
              t('chart.yAxisLabel', { unit: 'mg/dL' }),
            ]}
            labelFormatter={(label) => label}
          />
          <ReferenceArea y1={70} y2={140} fill="#16A34A" fillOpacity={0.08} />
          <ReferenceLine
            y={70}
            stroke="#DC2626"
            strokeDasharray="4 3"
            strokeWidth={1.5}
            label={{ value: '70', position: 'insideTopLeft', fill: '#DC2626', fontSize: 10 }}
          />
          <ReferenceLine
            y={140}
            stroke="#EAB308"
            strokeDasharray="4 3"
            strokeWidth={1.5}
            label={{ value: '140', position: 'insideTopLeft', fill: '#EAB308', fontSize: 10 }}
          />
          <ReferenceLine
            y={180}
            stroke="#DC2626"
            strokeDasharray="4 3"
            strokeWidth={1.5}
            label={{ value: '180', position: 'insideTopLeft', fill: '#DC2626', fontSize: 10 }}
          />
          <Line
            type="monotone"
            dataKey="glucoseMgDl"
            stroke="#2563EB"
            strokeWidth={2.5}
            dot={(props: any) => {
              const color = getStrokeColor(props.payload?.glucoseMgDl);
              return (
                <circle
                  cx={props.cx}
                  cy={props.cy}
                  r={4}
                  fill={color}
                  stroke="#fff"
                  strokeWidth={2}
                />
              );
            }}
            activeDot={{ r: 6, strokeWidth: 2, stroke: '#fff' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
