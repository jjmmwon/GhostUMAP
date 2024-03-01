import * as d3 from "d3";

export default function generateEdge(
  anomalyArray: [number, number][],
  hubpoint: [number, number],
  cluster: number[],
  centroids: [number, number][],
  xScale: d3.ScaleLinear<number, number, never>,
  yScale: d3.ScaleLinear<number, number, never>
) {
  centroids.forEach((d, i) => {
    centroids[i] = [(d[0] * 3 + hubpoint[0]) / 4, (d[1] * 3 + hubpoint[1]) / 4];
  });

  const edges: [number, number][][] = [];

  anomalyArray.forEach((point: [number, number], i) => {
    edges[i] = [point, centroids[cluster[i]], hubpoint];
  });

  const line = d3
    .line()
    .curve(d3.curveBundle.beta(1))
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]));

  return { edges, line };
}
