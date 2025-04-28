"use client";

import React, { useState, useRef } from "react";

export default function VoiceRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [botReply, setBotReply] = useState("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunks: Blob[] = [];

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");

      try {
        const res = await fetch("http://localhost:8080/voice", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        setTranscription(data.transcribed_text || "");
        setBotReply(data.bot_reply || "");
      } catch (err) {
        console.error("Error:", err);
      }
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="p-4 border rounded-lg max-w-md mx-auto space-y-4">
      <button
        className={`px-4 py-2 rounded ${isRecording ? "bg-red-500" : "bg-green-500"} text-white`}
        onClick={isRecording ? stopRecording : startRecording}
      >
        {isRecording ? "Stop Recording" : "Start Recording"}
      </button>

      {transcription && (
        <div>
          <p><strong>You said:</strong> {transcription}</p>
          <p><strong>Bot:</strong> {botReply}</p>
        </div>
      )}
    </div>
  );
}
