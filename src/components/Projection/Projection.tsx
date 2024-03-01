import { Box } from "@chakra-ui/react";

import { measureXScale, measureYScale } from "./Modules";

import CircleGroup from "./CircleGroup";
import AnomalyGroup from "./AnomalyGroup";
import { IData, IPoint, IProjection } from "../../types";

interface IProp {
  data: IData;
  width: number;
  height: number;
  margin: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

export default function Projection(props: IProp) {
  const { data, width, height, margin } = props;
  console.log(data);

  const projection: IPoint[] = data.projections[0].projection;
  const label: number[] = data.label;
  const anomaliesRank: number[] = data.anomaliesRank;

  const xScale = measureXScale(projection, width);

  const yScale = measureYScale(projection, height);

  const anomalies = anomaliesRank.slice(0, 3);

  return (
    <Box>
      <Box m="1rem">
        <svg
          id="projection-container"
          width={width + margin.left + margin.right}
          height={height + margin.top + margin.bottom}
        >
          <rect width="100%" height="100%" fill="white"></rect>
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            <CircleGroup
              projection={projection}
              xScale={xScale}
              yScale={yScale}
              label={label}
            />
            {
              // AnomalyGroup
              anomalies.map(function (anomaly: number, i: number) {
                const anomalyPositions: IPoint[] = data.projections.map(
                  (projection: IProjection) => {
                    return projection.projection[anomaly];
                  }
                );
                return (
                  <AnomalyGroup
                    key={i}
                    anomalyPositions={anomalyPositions}
                    xScale={xScale}
                    yScale={yScale}
                    k={3}
                  />
                );
              })
            }
          </g>
        </svg>
      </Box>
    </Box>
  );
}
