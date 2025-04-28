"use client"

import React, { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Mic, Send, MicOff, Bot, Sun, Moon, Menu, BookOpen, Calculator, Calendar, Users, Info } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { motion, AnimatePresence } from "framer-motion"
import { useTheme } from "next-themes"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import Link from "next/link"
import { useIsMobile } from "@/hooks/use-mobile"
import { v4 as uuidv4 } from "uuid"

type Message = {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
}

type SuggestedQuery = {
  id: string
  text: string
  icon?: React.ReactNode
}

export default function ChatInterface() {
  // State
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content:
        "Hello! I'm SiS Super Bot from Addis Ababa University's School of Information Science. How can I assist you today?",
      role: "assistant",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [micPulse, setMicPulse] = useState<number[]>([])
  const [micOn, setMicOn] = useState(false)
  const [error, setError] = useState<string>("")
  const [mounted, setMounted] = useState(false)

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const micAnimationRef = useRef<NodeJS.Timeout | null>(null)

  // Hooks
  const { theme, setTheme } = useTheme()
  const isMobile = useIsMobile()

  // Suggested queries
  const suggestedQueries: SuggestedQuery[] = [
    { id: "1", text: "Tell me about the IS program at AAU", icon: <BookOpen className="h-3 w-3" /> },
    { id: "2", text: "What is operational cost?", icon: <Users className="h-3 w-3" /> },
    { id: "3", text: "How do I calculate my GPA?", icon: <Calculator className="h-3 w-3" /> },
    { id: "4", text: "What is an IR System", icon: <BookOpen className="h-3 w-3" /> },
    { id: "5", text: "What is Concurrency Control?", icon: <Calendar className="h-3 w-3" /> },
  ]

  // Effects
  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    return () => {
      if (micAnimationRef.current) clearInterval(micAnimationRef.current)
    }
  }, [])

  // Handlers
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const handleSend = async (message?: string) => {
    const msgToSend = typeof message === "string" ? message : input
    if (!msgToSend.trim()) return

    const userMessage: Message = {
      id: uuidv4(),
      role: "user",
      content: msgToSend,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)
    setIsTyping(true)

    try {
      const response = await fetch("http://127.0.0.1:8080/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: userMessage.content }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to process the message.")
      }

      const data = await response.json()

      const botMessage: Message = {
        id: uuidv4(),
        role: "assistant",
        content: data.bot_reply || "Sorry, I couldn’t understand.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage])
    } catch (error) {
      console.error("Error sending message:", error)
      setMessages((prev) => [
        ...prev,
        {
          id: uuidv4(),
          content: "There was an error connecting to the server.",
          role: "assistant",
          timestamp: new Date(),
        },
      ])
    } finally {
      setIsLoading(false)
      setIsTyping(false)
    }
  }

  const handleSuggestedQuery = (query: string) => {
    setInput(query)
    setTimeout(() => handleSend(), 100)
  }

  const toggleRecording = async () => {
    if (!isRecording) {
      setIsRecording(true)
      setMicPulse([])

      let stream: MediaStream

      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      } catch (err) {
        console.error("Microphone access denied:", err)
        setIsRecording(false)
        return
      }

      const mediaRecorder = new MediaRecorder(stream)
      const audioChunks: BlobPart[] = []

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" })
        console.log("Audio Blob Size:", audioBlob.size); // Debug log

        if (audioBlob.size === 0) {
          console.error("Audio Blob is empty. Recording failed.");
          setMessages((prev) => [
            ...prev,
            {
              id: uuidv4(),
              content: "Recording failed. Please try again.",
              role: "assistant",
              timestamp: new Date(),
            },
          ]);
          setIsRecording(false);
          return;
        }

        const formData = new FormData();
        formData.append("audio", audioBlob, "voiceInput.webm");

        console.log("FormData:", formData.get("audio")); // Debug log

        setIsTyping(true);
        setIsLoading(true);

        try {
          const response = await fetch("http://127.0.0.1:8080/voice", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Failed to process voice message");
          }

          const data = await response.json();
          console.log("Response data:", data);

          const botMessage: Message = {
            id: uuidv4(),
            content: data.bot_reply || "Sorry, I couldn’t understand.",
            role: "assistant",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, botMessage]);
        } catch (err) {
          console.error("Voice processing error:", err);
          setMessages((prev) => [
            ...prev,
            {
              id: uuidv4(),
              content: "There was an error processing your voice message.",
              role: "assistant",
              timestamp: new Date(),
            },
          ]);
        } finally {
          setIsTyping(false);
          setIsLoading(false);
        }
      }

      mediaRecorder.start()

      micAnimationRef.current = setInterval(() => {
        setMicPulse(Array.from({ length: 5 }, () => Math.random() * 40 + 10))
      }, 150)

      setTimeout(() => {
        mediaRecorder.stop()
        stream.getTracks().forEach((track) => track.stop())

        if (micAnimationRef.current) {
          clearInterval(micAnimationRef.current)
          micAnimationRef.current = null
        }

        setIsRecording(false)
      }, 4000)
    }
  }

  const handleVoiceInput = async () => {
    try {
      setMicOn(true)
      setError("")
      setIsLoading(true)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      const chunks: Blob[] = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data)
      }

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" })
        const formData = new FormData()
        formData.append("audio", blob, "recording.webm")

        try {
          const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData,
          })

          const data = await response.json()
          const transcription = data.transcription?.trim()

          if (!transcription) throw new Error("Could not understand audio")

          setInput(transcription)
          await handleSend(transcription)
        } catch (err) {
          setError("Transcription failed. Please try again.")
        } finally {
          setIsLoading(false)
          setMicOn(false)
        }
      }

      mediaRecorder.start()
      setTimeout(() => mediaRecorder.stop(), 4000)
    } catch (err) {
      console.error("Voice input error:", err)
      setError("Microphone access denied or unavailable.")
      setIsLoading(false)
      setMicOn(false)
    }
  }

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  // Quick access links for desktop
  const quickAccessLinks = (
    <div className="space-y-2">
      <Link href="/department-info" className="w-full">
        <Button variant="ghost" className="w-full justify-start" size="sm">
          <Info className="mr-2 h-4 w-4" />
          Department Info
        </Button>
      </Link>
      <Link href="/program-details" className="w-full">
        <Button variant="ghost" className="w-full justify-start" size="sm">
          <BookOpen className="mr-2 h-4 w-4" />
          Program Details
        </Button>
      </Link>
      <Link href="/faculty" className="w-full">
        <Button variant="ghost" className="w-full justify-start" size="sm">
          <Users className="mr-2 h-4 w-4" />
          Faculty & Staff
        </Button>
      </Link>
      <Link href="/gpa-calculator" className="w-full">
        <Button variant="ghost" className="w-full justify-start" size="sm">
          <Calculator className="mr-2 h-4 w-4" />
          GPA Calculator
        </Button>
      </Link>
    </div>
  )

  // Render
  return (
    <div className="flex flex-col h-[calc(100vh-12rem)] md:h-[70vh] w-full max-w-4xl mx-auto rounded-xl shadow-lg relative overflow-hidden bg-card border border-border transition-colors duration-300">
      <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none"></div>
      <div className="p-3 md:p-4 border-b border-border flex items-center justify-between bg-muted/30 relative z-10">
        <div className="flex items-center">
          <Avatar className="h-8 w-8 mr-2">
            <AvatarImage src="/images/sis-logo.png" alt="SiS Super Bot" />
            <AvatarFallback className="bg-primary text-primary-foreground">SiS</AvatarFallback>
          </Avatar>
          <div>
            <h3 className="font-semibold text-sm md:text-base">SiS Super Bot</h3>
            <div className="flex items-center">
              <span className="h-2 w-2 rounded-full bg-green-500 mr-1.5"></span>
              <span className="text-xs text-muted-foreground">Online</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {mounted && (
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 rounded-full transition-colors duration-300 hover:bg-muted"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            >
              {theme === "dark" ? (
                <Sun className="h-4 w-4 text-yellow-400" />
              ) : (
                <Moon className="h-4 w-4 text-slate-700" />
              )}
            </Button>
          )}
          {!isMobile && (
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full hover:bg-muted">
                  <Menu className="h-4 w-4" />
                  <span className="sr-only">Menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[250px]">
                <div className="py-6 space-y-6">
                  <h3 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground">Quick Access</h3>
                  {quickAccessLinks}
                </div>
              </SheetContent>
            </Sheet>
          )}
        </div>
      </div>
      <ScrollArea className="flex-1 p-3 md:p-4 relative">
        <div className="space-y-4 pb-4">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isTyping && (
            <div className="flex items-start gap-3">
              <Avatar className="h-8 w-8">
                <AvatarImage src="/images/sis-logo.png" alt="SiSChatBot" />
                <AvatarFallback className="bg-primary text-primary-foreground">SiS</AvatarFallback>
              </Avatar>
              <div className="rounded-md bg-muted p-3 text-sm">
                <div className="flex space-x-2">
                  <div className="h-2 w-2 rounded-full bg-primary animate-pulse"></div>
                  <div className="h-2 w-2 rounded-full bg-primary animate-pulse delay-150"></div>
                  <div className="h-2 w-2 rounded-full bg-primary animate-pulse delay-300"></div>
                </div>
              </div>
            </div>
          )}
          {messages.length === 1 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="pt-4"
            >
              <h4 className="text-sm font-medium mb-2">Suggested questions:</h4>
              <div className="flex flex-wrap gap-2">
                {suggestedQueries.map((query) => (
                  <Button
                    key={query.id}
                    variant="outline"
                    size="sm"
                    className="text-xs bg-muted/50 border-primary/20 hover:bg-primary/10 h-auto py-2 px-3 touch-manipulation"
                    onClick={() => handleSuggestedQuery(query.text)}
                  >
                    {query.icon && <span className="mr-1.5">{query.icon}</span>}
                    {query.text}
                  </Button>
                ))}
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>
      <div className="p-3 md:p-4 border-t border-border bg-card transition-colors duration-300">
        <div className="flex items-center space-x-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about courses, faculty, events..."
            className="border-primary/20 focus-visible:ring-primary h-10 md:h-11 text-sm md:text-base bg-background transition-colors duration-300"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />
          <AnimatePresence mode="wait">
            {isRecording ? (
              <motion.div
                key="recording"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                className="relative"
              >
                <Button
                  variant="outline"
                  size="icon"
                  className="border-primary text-primary relative z-10 h-10 w-10 md:h-11 md:w-11 touch-manipulation transition-colors duration-300"
                  onClick={toggleRecording}
                >
                  <Mic className="h-4 w-4 md:h-5 md:w-5" />
                </Button>
                <motion.div
                  className="absolute -inset-2 rounded-full bg-primary/10 z-0"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY }}
                ></motion.div>
                <div className="absolute -bottom-12 left-1/2 transform -translate-x-1/2 flex items-end h-8 space-x-1 bg-background/80 px-2 py-1 rounded-md">
                  {micPulse.map((height, i) => (
                    <motion.div
                      key={i}
                      className="w-1 bg-primary rounded-full"
                      style={{ height: `${height}%` }}
                      animate={{ height: `${height}%` }}
                      transition={{ duration: 0.2 }}
                    ></motion.div>
                  ))}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="not-recording"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
              >
                <Button
                  variant="outline"
                  size="icon"
                  className="border-primary/20 hover:bg-primary/10 hover:text-primary hover:border-primary transition-colors duration-300 h-10 w-10 md:h-11 md:w-11 touch-manipulation"
                  onClick={toggleRecording}
                >
                  <MicOff className="h-4 w-4 md:h-5 md:w-5" />
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
          <Button
            variant="default"
            size="icon"
            className="bg-primary hover:bg-primary/90 relative overflow-hidden group h-10 w-10 md:h-11 md:w-11 touch-manipulation"
            onClick={() => handleSend()}
            disabled={!input.trim()}
          >
            <span className="relative z-10">
              <Send className="h-4 w-4 md:h-5 md:w-5" />
            </span>
            <span className="absolute inset-0 h-full w-0 bg-gradient-to-r from-primary to-secondary/80 transition-all duration-300 group-hover:w-full"></span>
          </Button>
        </div>
        <div className="mt-2 text-xs text-muted-foreground flex flex-col md:flex-row md:items-center justify-between">
          <span className="hidden md:inline">
            {isRecording ? (
              <span className="text-primary">Recording... Speak now</span>
            ) : (
              <span>Press Enter to send, Shift+Enter for new line</span>
            )}
          </span>
          <span className="flex items-center text-[10px] md:text-xs">
            <Bot className="h-3 w-3 mr-1" />
            Powered by SiS Super Bot | <span className="ml-1 text-primary">IS Seer</span>
          </span>
        </div>
      </div>
    </div>
  )
  console.log("Messages array:", messages);
}
