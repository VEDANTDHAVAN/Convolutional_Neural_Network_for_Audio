import Link from "next/link";

export default function HomePage() {
  return (
    <main className="flex min-h-screen p-8 flex-col items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
      <div className="mx-auto max-w-[70%]">
       <div className="mb-12 text-center">
        <h1 className="mb-4 text-4xl font-semibold tracking-light">Audio Visualizer using Convolutional Neural Network</h1>
        <p className="mb-8 text-lg text-white/70">Upload a WAV file to see the Model's Predictions and Feature Maps</p>
        <div className="flex flex-col items-center">
         <div className="relative inline-block">
          <input type="file" accept=".wav" id="file-upload" className="absolute inset-0 w-full cursor-pointer opacity-0" />
         </div>
        </div>
       </div>
      </div>
    </main>
  );
}
