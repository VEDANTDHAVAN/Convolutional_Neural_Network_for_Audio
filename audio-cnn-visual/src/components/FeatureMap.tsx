import { getColor } from "~/lib/colors";

const FeatureMap = ({data, title, internal}:{
    data: number[][], title: string, internal?: boolean
}) => {
  if (!data?.length || !data[0]?.length) return null;
 
  const mapHeight = data.length;
  const mapWidth = data[0].length;
 
  const absMax = data
    .flat() //For each number we calculate the absolute value, then we compare it to the current maximum
    .reduce((acc, val) => Math.max(acc, Math.abs(val ?? 0)) , 0); 

  return (
    <>
     <div className="w-full text-center">
      <svg 
      viewBox={`0 0 ${mapWidth} ${mapHeight}`}
      preserveAspectRatio="none" 
      className={`mx-auto block rounded border border-stone-400 ${internal ? "w-full max-w-32":"w-full max-w-[500px] max-h-[300px] object-contain"}`}
      >
      {data.flatMap((row, i) => row.map((val, j) => {
        const normalizedValue = absMax === 0 ? 0 : val / absMax;
        const [r, g, b] = getColor(normalizedValue);
        return (
            <rect key={`${i} - ${j}`} x={j} y={i} width={1} height={1} fill={`rgb(${r}, ${g}, ${b})`} />
        )
      }))}
      </svg>
      <p className="mt-2 text-sm text-gray-700">{title}</p>
     </div>
    </>
  )
} 

export default FeatureMap;