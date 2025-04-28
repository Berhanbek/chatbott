"use client"

import { Button } from "@/components/ui/button"
import { Calculator, HelpCircle, Info, BookOpen, Users } from "lucide-react"

interface SidebarNavProps {
  activeTab: "info" | "faq" | "gpa" | "program" | "faculty"
  setActiveTab: (tab: "info" | "faq" | "gpa" | "program" | "faculty") => void
}

export function SidebarNav({ activeTab, setActiveTab }: SidebarNavProps) {
  return (
    <div className="flex flex-col space-y-1 p-2">
      <Button
        variant={activeTab === "info" ? "secondary" : "ghost"}
        className="justify-start terminal-text"
        onClick={() => setActiveTab("info")}
      >
        <Info className="mr-2 h-4 w-4" />
        Department Info
      </Button>
      <Button
        variant={activeTab === "program" ? "secondary" : "ghost"}
        className="justify-start terminal-text"
        onClick={() => setActiveTab("program")}
      >
        <BookOpen className="mr-2 h-4 w-4" />
        Program Details
      </Button>
      <Button
        variant={activeTab === "faculty" ? "secondary" : "ghost"}
        className="justify-start terminal-text"
        onClick={() => setActiveTab("faculty")}
      >
        <Users className="mr-2 h-4 w-4" />
        Faculty & Staff
      </Button>
      <Button
        variant={activeTab === "faq" ? "secondary" : "ghost"}
        className="justify-start terminal-text"
        onClick={() => setActiveTab("faq")}
      >
        <HelpCircle className="mr-2 h-4 w-4" />
        FAQ
      </Button>
      <Button
        variant={activeTab === "gpa" ? "secondary" : "ghost"}
        className="justify-start terminal-text"
        onClick={() => setActiveTab("gpa")}
      >
        <Calculator className="mr-2 h-4 w-4" />
        GPA Calculator
      </Button>
    </div>
  )
}
