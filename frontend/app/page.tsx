"use client"

import { useState, useEffect } from "react"
import ChatInterface from "@/components/chat-interface"
import { Button } from "@/components/ui/button"
import { BookOpen, Users, Calculator, Info, Menu, Mic, MicOff } from "lucide-react"
import Link from "next/link"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { useIsMobile } from "@/hooks/use-mobile"
import { motion } from "framer-motion"

export default function Home() {
  const [showWelcome, setShowWelcome] = useState(true)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const isMobile = useIsMobile()
  const [mounted, setMounted] = useState(false)

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)] bg-background">
      {showWelcome && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-primary/10 to-secondary/10 py-3 px-4 text-center relative"
        >
          <p className="text-sm">
            Welcome to <strong>SiS Super Bot</strong> - Your AI assistant for the School of Information Science
          </p>
          <button
            onClick={() => setShowWelcome(false)}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            aria-label="Dismiss welcome message"
          >
            ×
          </button>
        </motion.div>
      )}

      <div className="flex flex-1 flex-col relative">
        {/* Main Chat Section */}
        <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full px-2 sm:px-4 md:px-6 py-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-4 text-center"
          >
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-2">
              <span className="gold-gradient">SiS Super Bot</span>
            </h1>
            <p className="text-muted-foreground text-sm md:text-base max-w-2xl mx-auto">
              Your AI assistant for the School of Information Science
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="flex-1 flex flex-col"
          >
            <ChatInterface />
          </motion.div>
        </div>
        <Button variant="ghost" size="sm" onClick={() => setVoiceEnabled(prev => !prev)}>
  {voiceEnabled ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
  {voiceEnabled ? "Disable Voice" : "Enable Voice"}
</Button>

        {/* Mobile Quick Access Menu */}
        {isMobile && (
          <Sheet>
            <SheetTrigger asChild>
              <Button
                size="icon"
                className="fixed bottom-6 right-6 h-12 w-12 rounded-full shadow-lg bg-primary hover:bg-primary/90 z-50"
              >
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[250px]">
              <div className="py-6 space-y-6">
                <h3 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground">
                  Quick Access
                </h3>
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
              </div>
            </SheetContent>
          </Sheet>
        )}
      </div>
    </div>
  )
}
