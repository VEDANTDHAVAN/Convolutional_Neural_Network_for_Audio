/* eslint-disable @typescript-eslint/consistent-indexed-object-style */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/prefer-nullish-coalescing */
"use client";

import React, { useState } from "react";
import ColorScale from "~/components/ColorScale";
import FeatureMap from "~/components/FeatureMap";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";
import WaveForm from "~/components/WaveForm";

// Created types for API Integration
interface Prediction {
  class: string,
  confidence: number,
}

interface LayerData {
  shape: number[],
  values: number[][],
}

interface VisualData {
  [layerName: string]: LayerData,
}

interface WaveFormData {
  values: number[],
  sample_rate: number,
  duration: number,
}

interface APIResponse {
  predictions: Prediction[],
  visualization: VisualData,
  input_spectogram: LayerData,
  waveform: WaveFormData,
}

const ESC50_EMOJI_MAP: Record<string, string> = {
  dog: "🐕",
  rain: "🌧️",
  crying_baby: "👶",
  door_wood_knock: "🚪",
  helicopter: "🚁",
  rooster: "🐓",
  sea_waves: "🌊",
  sneezing: "🤧",
  mouse_click: "🖱️",
  chainsaw: "🪚",
  pig: "🐷",
  crackling_fire: "🔥",
  clapping: "👏",
  keyboard_typing: "⌨️",
  siren: "🚨",
  cow: "🐄",
  crickets: "🦗",
  breathing: "💨",
  door_wood_creaks: "🚪",
  car_horn: "📯",
  frog: "🐸",
  chirping_birds: "🐦",
  coughing: "😷",
  can_opening: "🥫",
  engine: "🚗",
  cat: "🐱",
  water_drops: "💧",
  footsteps: "👣",
  washing_machine: "🧺",
  train: "🚂",
  hen: "🐔",
  wind: "💨",
  laughing: "😂",
  vacuum_cleaner: "🧹",
  church_bells: "🔔",
  insects: "🦟",
  pouring_water: "🚰",
  brushing_teeth: "🪥",
  clock_alarm: "⏰",
  airplane: "✈️",
  sheep: "🐑",
  toilet_flush: "🚽",
  snoring: "😴",
  clock_tick: "⏱️",
  fireworks: "🎆",
  crow: "🐦‍⬛",
  thunderstorm: "⛈️",
  drinking_sipping: "🥤",
  glass_breaking: "🔨",
  hand_saw: "🪚",
};

const getEmojiForEachClass = (className: string): string => {
  return ESC50_EMOJI_MAP[className] || "💡";
}

//fuction to split Layers
function splitLayers(visualization: VisualData) {
  const main: [string, LayerData][] = [];
  const internals: Record<string, [string, LayerData][]> = {};

  for (const [name, data] of Object.entries(visualization)){
    if (!name.includes(".")){
      main.push([name, data]);
    } else {
      const [parent] = name.split("."); //[] destructuring helps to select the first part after split
      if (parent === undefined) continue;

      if(!internals[parent]) internals[parent] = [];
      internals[parent].push([name, data]);
    }
  } 

  return { main, internals }
}

export default function HomePage() {
  const [isloading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [visualData, setVisualData] = useState<APIResponse | null>(null);
 
  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVisualData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        //First get the array buffer
      const arrBuffer = reader.result as ArrayBuffer;
      //then convert it into base64 String
      const base64String = btoa(
        new Uint8Array(arrBuffer).reduce(
        (data, byte) => data + String.fromCharCode(byte), //this converts a byte's number into a CharCode and adds it to the String
         ""),
        ); 
      
      const response = await fetch(
        "https://vedantdhavan--audio-cnn-inference-audioclassifier-inference.modal.run/",
        {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({audio_data: base64String}),
        },
      );

      if (!response.ok){
        throw new Error(`API Error ${response.statusText}`)
      }

      const data: APIResponse = await response.json();
      setVisualData(data);
      } catch (error) {
        setError(error instanceof Error ? error.message : "An Unknown Error occured!!");
      } finally {
        setIsLoading(false);
      }
    };
    reader.onerror = () => {
      setError("Falied to read and extract Data from the File.");
      setIsLoading(false);
    }
  };

  const {main, internals} = visualData ? splitLayers(visualData?.visualization) : {main: [], internals: {}}

  return (
    <main className="flex min-h-screen p-8 flex-col items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
      <div className="mx-auto max-w-[100%]">
       <div className="mb-12 text-center">
        <h1 className="mb-4 text-4xl font-semibold tracking-light">Audio Visualizer using Convolutional Neural Network</h1>
        <p className="mb-8 text-lg text-white/70">Upload a WAV file to see the Model&apos;s Predictions and Feature Maps</p>
        <div className="flex flex-col items-center">
         <div className="relative inline-block">
          <input type="file" accept=".wav" disabled={isloading} id="file-upload" onChange={handleFileChange} className="absolute inset-0 w-full cursor-pointer opacity-0" />
          <Button disabled={isloading} variant="outline"size="lg" className="text-black">{isloading ? "Analysing Audio File....":"Choose File" }</Button>
         </div>
         {fileName && (<Badge variant="secondary" className="mt-5 bg-stone-200 text-green-700">{fileName}</Badge>)}
        </div>
       </div>
       {error && (<Card className="mb-8 border-red-600 bg-red-50">
        <CardContent>
          <p className="text-red-600">Error: {error}</p>
        </CardContent>
       </Card>)}

       {true && (<div className="space-y-8">
        <Card>
          <CardHeader className="text-3xl font-bold text-green-700">Top 3 Predictions</CardHeader>
          <CardContent>
            <div className="space-y-4">
              {visualData?.predictions.slice(0,3).map((pred, idx) => (
                <div key={pred.class} className="space-y-2">
                  <div className="flex items-center justify-between">
                   <div className="">
                     {getEmojiForEachClass(pred.class)}{" "}
                   <span className="text-2xl font-semibold text-purple-900">{pred.class.replaceAll("_", "")}</span> 
                   </div>
                   <Badge variant={idx === 0 ? "default":"secondary"}>{(pred.confidence * 100).toFixed(1)}%</Badge>
                  </div>
                  <Progress value={pred.confidence * 100} className="h-2 bg-gray-200"/>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
         <Card>
          <CardHeader>
            <CardTitle className="text-3xl font-bold text-blue-700">Input Spectogram</CardTitle>
            </CardHeader>
          <CardContent>
            {/* Feature Maps */}
             <FeatureMap data={visualData?.input_spectogram.values} spectogram
              title={`${visualData?.input_spectogram.shape.join(" x ")}`}/>
            {/* Color Scale */}
            <div className="mt-5 flex justify-end">
              <ColorScale width={200} height={16} min={-1} max={1} />
            </div>
          </CardContent>
         </Card>

         <Card>
          <CardHeader><CardTitle className="text-3xl font-bold text-purple-700">Audio Waveform</CardTitle></CardHeader>
          <CardContent>
           <WaveForm data={visualData?.waveform.values} title={`${visualData?.waveform.duration.toFixed(2)}s * ${visualData?.waveform.sample_rate}Hz`} /> 
          </CardContent>
         </Card>
        </div>
        {/* Feature Maps */}
        <Card>
          <CardHeader>
            <CardTitle className="text-3xl font-bold text-orange-700">Convolutional Layer Outputs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 gap-6">
              {main.map(([mainName, mainData]) => (
                <div key={mainName} className="space-y-4">
                 <div>
                  <h4 className="mb-2 font-medium text-gray-900">{mainName}</h4>
                  <FeatureMap data={mainData.values} title={`${mainData.shape.join(" x ")}`}/>
                 </div>
                 {internals[mainName] && <div className="h-80 overflow-y-auto rounded border border-gray-300 bg-stone-50 p-2">
                   <div className="space-y-2">
                    {internals[mainName].sort(([a], [b]) => a.localeCompare(b))
                     .map(([layerName, layerData]) => (
                     <FeatureMap key={layerName} data={layerData.values} 
                     title={layerName.replace(`${mainName}.`, "")} internal={true} />))
                    } 
                   </div>
                  </div>}
                </div>
              ))}
            </div>
            <div className="mt-5 flex justify-end">
              <ColorScale width={200} height={16} min={-1} max={1} />
            </div>
          </CardContent>
        </Card>
       </div>
      )}
      </div>
    </main>
  );
}
