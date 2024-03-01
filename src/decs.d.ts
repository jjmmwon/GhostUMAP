declare module "skmeans" {
  export default function skmeans(
    data: number[][],
    k: number
  ): { it: number; k: number; idxs: number[]; centroids: [number, number][] };
}
