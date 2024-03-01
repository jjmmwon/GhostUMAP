import * as d3 from "d3";
import { IPoint } from "../../../types";

const measureXScale = (projection: IPoint[], width: number) => {
  const xExtent: [number, number] = d3.extent<number>(
    projection.map((d: IPoint) => Number(d.x))
  ) as [number, number];
  const [xMin, xMax] = [xExtent[0], xExtent[1]];
  console.log(xMin, xMax);

  return d3
    .scaleLinear()
    .domain([xMin * 1.05, xMax * 1.05])
    .range([0, width]);
};
const measureYScale = (projection: IPoint[], height: number) => {
  const yExtent: [number, number] = d3.extent(
    projection.map((d: IPoint) => Number(d.y))
  ) as [number, number];
  const [yMin, yMax]: number[] = [yExtent[0], yExtent[1]];
  console.log(yMin, yMax);

  return d3
    .scaleLinear()
    .domain([yMin * 1.05, yMax * 1.05])
    .range([height, 0]);
};

export { measureXScale, measureYScale };
