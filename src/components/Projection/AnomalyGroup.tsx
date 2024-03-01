import { ScaleLinear, symbol, symbolStar } from "d3";
import { IPoint } from "../../types";
import skmeans from "skmeans";
import { generateEdge } from "./Modules";

interface IProp {
  anomalyPositions: IPoint[];
  xScale: ScaleLinear<number, number, never>;
  yScale: ScaleLinear<number, number, never>;
  k: number;
}

export default function AnomalyGroup(props: IProp) {
  const { anomalyPositions, xScale, yScale, k } = props;
  console.log(anomalyPositions, k);

  const anomalies = anomalyPositions.map((d: IPoint, i: number) => (
    <circle key={i} cx={xScale(d.x)} cy={yScale(d.y)} r={5} fill="black" />
  ));

  const anomalyArray = anomalyPositions.map((d: IPoint) => [
    Number(d.x),
    Number(d.y),
  ]) as [number, number][];
  const kMeansResult = skmeans(anomalyArray, 3);

  // const centroids = kMeansResult.centroids.map((d: number[], i: number) => (
  //   <path
  //     key={i}
  //     d={symbol().type(symbolTriangle).size(100)() as string}
  //     stroke="#FFCE44"
  //     fill="#FFCE44"
  //     transform={`translate(${xScale(d[0])}, ${yScale(d[1])})`}
  //   />
  // ));

  const hubpoint: [number, number] = [
    Math.random() * 22 - 5,
    Math.random() * 20 - 5,
  ];
  const hubpointStar = (
    <path
      d={symbol().type(symbolStar).size(10)() as string}
      stroke="black"
      fill="#FFCE44"
      transform={`translate(${xScale(hubpoint[0])}, ${yScale(hubpoint[1])})`}
    />
  );

  const { edges, line } = generateEdge(
    anomalyArray,
    hubpoint,
    kMeansResult.idxs,
    kMeansResult.centroids,
    xScale,
    yScale
  );

  const lines = edges.map((d: [number, number][], i: number) => (
    <path
      key={i}
      d={line(d) || ""}
      stroke="#FFCE44"
      fill="none"
      strokeWidth={1}
      opacity={0.5}
    />
  ));

  return (
    <g>
      <g>{anomalies}</g>
      {/* <g>{centroids}</g> */}
      <g>{lines}</g>
      <g>{hubpointStar}</g>
    </g>
  );
}
