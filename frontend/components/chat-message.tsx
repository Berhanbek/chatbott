"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"
import { motion } from "framer-motion"

type Message = {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
}

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isBot = message.role === "assistant"

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("flex items-start gap-2 md:gap-3 touch-manipulation", isBot ? "justify-start" : "justify-end")}
    >
      {isBot && (
        <Avatar className="h-7 w-7 md:h-8 md:w-8 flex-shrink-0">
          <AvatarImage src="/images/sis-logo.png" alt="SiS Super Bot" />
          <AvatarFallback className="bg-primary text-primary-foreground">SiS</AvatarFallback>
        </Avatar>
      )}
      <div
        className={cn(
          "rounded-2xl p-2 md:p-3 text-xs md:text-sm max-w-[85%] transition-colors duration-300",
          isBot ? "bg-muted text-foreground rounded-tl-none" : "bg-primary text-primary-foreground rounded-tr-none",
        )}
      >
        <div className="mb-1 text-[10px] md:text-xs text-muted-foreground flex justify-between">
          <span>{isBot ? "SiS Super Bot" : "You"}</span>
          <span className="ml-2">{formatTime(message.timestamp)}</span>
        </div>
        <div className="whitespace-pre-wrap break-words">{message.content}</div>
      </div>
      {!isBot && (
        <Avatar className="h-7 w-7 md:h-8 md:w-8 flex-shrink-0">
          <AvatarFallback className="bg-secondary/20 text-secondary-foreground">U</AvatarFallback>
        </Avatar>
      )}
    </motion.div>
  )
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
}
