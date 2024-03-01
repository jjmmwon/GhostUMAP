import * as d3 from "d3";
import { IPoint } from "../../types";
import { useEffect, useState } from "react";

interface IProp {
  projection: IPoint[];
  xScale: d3.ScaleLinear<number, number, never>;
  yScale: d3.ScaleLinear<number, number, never>;
  label: string[] | number[];
}

export default function CircleGroup(props: IProp) {
  const { projection, xScale, yScale, label } = props;
  const [circles, setCircles] = useState<JSX.Element[]>([]);

  useEffect(() => {
    const newCircles = projection.map((d: IPoint, i: number) => (
      <circle
        key={i}
        cx={xScale(Number(d.x))}
        cy={yScale(Number(d.y))}
        r={1}
        opacity={0.7}
        fill={d3.schemeTableau10[Number(label[i])]}
      />
    ));
    setCircles(newCircles);
  }, [projection, xScale, yScale, label]);

  return <g>{circles}</g>;
}
