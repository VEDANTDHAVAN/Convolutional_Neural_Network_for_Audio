"use client";

import Link from "next/link";
import React, { useState } from "react";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";

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
  prediction: Prediction[],
  visualization: VisualData,
  input_spectogram: LayerData,
  waveform: WaveFormData,
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
  }

  return (
    <main className="flex min-h-screen p-8 flex-col items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
      <div className="mx-auto max-w-[70%]">
       <div className="mb-12 text-center">
        <h1 className="mb-4 text-4xl font-semibold tracking-light">Audio Visualizer using Convolutional Neural Network</h1>
        <p className="mb-8 text-lg text-white/70">Upload a WAV file to see the Model's Predictions and Feature Maps</p>
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
      </div>
    </main>
  );
}
