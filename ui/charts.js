import {Chart, registerables} from 'https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.esm.js';

Chart.register(...registerables);

Chart.defaults.color = '#d7dcff';
Chart.defaults.font.family = 'Inter, "Segoe UI", Roboto, sans-serif';
Chart.defaults.font.size = 12;

const AXIS_TICK_COLOR = '#9ba8d6';
const AXIS_GRID_COLOR = 'rgba(139,92,246,0.18)';
const AXIS_BORDER_COLOR = 'rgba(139,92,246,0.28)';

const baseDataset = {
  fill: false,
  tension: 0.35,
  pointRadius: 0,
  pointHoverRadius: 4,
  pointHitRadius: 6,
  pointHoverBorderWidth: 0,
  borderWidth: 2.5,
  parsing: false,
  spanGaps: false,
};

const baseOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    mode: 'nearest',
    intersect: false,
  },
  animation: false,
  layout: {
    padding: 6,
  },
  scales: {
    x: {
      type: 'linear',
      grid: {
        color: AXIS_GRID_COLOR,
      },
      border: {
        color: AXIS_BORDER_COLOR,
      },
      ticks: {
        color: AXIS_TICK_COLOR,
        maxRotation: 0,
        autoSkipPadding: 14,
      },
    },
    y: {
      grid: {
        color: AXIS_GRID_COLOR,
      },
      border: {
        color: AXIS_BORDER_COLOR,
      },
      ticks: {
        color: '#e7ebff',
        padding: 6,
      },
    },
  },
  plugins: {
    legend: {
      labels: {
        color: '#e7ebff',
        usePointStyle: false,
        boxWidth: 18,
        padding: 16,
      },
    },
    tooltip: {
      backgroundColor: 'rgba(15,20,45,0.92)',
      borderColor: 'rgba(139,92,246,0.45)',
      borderWidth: 1,
      titleColor: '#f8f9ff',
      bodyColor: '#dfe3ff',
      cornerRadius: 10,
      displayColors: false,
      padding: 12,
    },
  },
  elements: {
    line: {
      borderCapStyle: 'round',
      borderJoinStyle: 'round',
    },
  },
};

function createLineChart(canvasId, datasets) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      datasets: datasets.map((dataset) => ({
        ...baseDataset,
        ...dataset,
        data: [],
      })),
    },
    options: baseOptions,
  });
}

export const rewardChart = createLineChart('rewardChart', [
  {
    label: 'Average Reward',
    borderColor: '#a56dff',
    backgroundColor: '#a56dff',
  },
  {
    label: 'Greedy Reward',
    borderColor: '#ffcc00',
    backgroundColor: '#ffcc00',
    borderDash: [5, 5],
  },
]);

export const fruitChart = createLineChart('fruitChart', [
  {
    label: 'Average Fruit',
    borderColor: '#00ffff',
    backgroundColor: '#00ffff',
  },
  {
    label: 'Greedy Fruit',
    borderColor: '#00cc66',
    backgroundColor: '#00cc66',
    borderDash: [5, 5],
  },
]);

export function resetProgressCharts() {
  [rewardChart, fruitChart].forEach((chart) => {
    if (!chart) return;
    chart.data.datasets.forEach((dataset) => {
      dataset.data = [];
    });
    chart.update('none');
  });
}

export function syncProgressCharts({
  progressPoints = [],
  greedyEpisodes = [],
  greedyRewards = [],
  greedyFruits = [],
  limit = 120,
} = {}) {
  if (rewardChart) {
    const rewardPoints = progressPoints
      .slice(-limit)
      .filter((point) => Number.isFinite(point?.episode) && Number.isFinite(point?.reward))
      .map((point) => ({ x: point.episode, y: point.reward }));
    const greedyRewardPoints = greedyEpisodes
      .slice(-limit)
      .map((episode, idx) => ({ episode, value: greedyRewards[idx] }))
      .filter((point) => Number.isFinite(point.episode) && Number.isFinite(point.value))
      .map((point) => ({ x: point.episode, y: point.value }));
    rewardChart.data.datasets[0].data = rewardPoints;
    rewardChart.data.datasets[1].data = greedyRewardPoints;
    rewardChart.update('none');
  }

  if (fruitChart) {
    const fruitPoints = progressPoints
      .slice(-limit)
      .filter((point) => Number.isFinite(point?.episode) && Number.isFinite(point?.fruit))
      .map((point) => ({ x: point.episode, y: point.fruit }));
    const greedyFruitPoints = greedyEpisodes
      .slice(-limit)
      .map((episode, idx) => ({ episode, value: greedyFruits[idx] }))
      .filter((point) => Number.isFinite(point.episode) && Number.isFinite(point.value))
      .map((point) => ({ x: point.episode, y: point.value }));
    fruitChart.data.datasets[0].data = fruitPoints;
    fruitChart.data.datasets[1].data = greedyFruitPoints;
    fruitChart.update('none');
  }
}
