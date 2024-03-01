interface IPoint {
  i: number;
  x: number;
  y: number;
}

interface IProjectionCSV {
  projection: string;
}

interface IProjection {
  projection: IPoint[];
}

interface IData {
  projections: IProjection[];
  label: number[];
  anomaliesRank: number[];
}

interface DataResponse {
  projections: IProjectionCSV[];
  label: number[];
  anomaliesRank: number[];
}

export type { IPoint, IProjection, IData, DataResponse };
