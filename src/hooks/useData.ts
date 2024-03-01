import { useEffect, useState } from "react";
import Papa from "papaparse";
import { DataResponse, IData, IPoint } from "../types";

export default function useData(link: string) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<IData | null>(null);

  useEffect(() => {
    async function fetchData() {
      const response = await fetch(link);
      const data: DataResponse = await response.json();
      console.log(data);

      const anomaliesRank = data.anomaliesRank;
      const projections = data.projections.map((projection) => {
        const parsedData: Papa.ParseResult<IPoint> = Papa.parse(
          projection.projection,
          {
            header: true,
          }
        );
        return {
          projection: parsedData.data,
        };
      });
      const label = data.label;

      console.log(projections);
      console.log(anomaliesRank);
      console.log(label);
      setData({ projections, label, anomaliesRank });
      setLoading(false);
    }
    setLoading(true);
    fetchData();
  }, [link]);
  console.log(data);
  return { loading, data };
}
