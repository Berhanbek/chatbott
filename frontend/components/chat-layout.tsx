"use client"

import type React from "react"

import { useState } from "react"
import { SidebarProvider, Sidebar, SidebarContent, SidebarHeader, SidebarFooter } from "@/components/ui/sidebar"
import { SidebarNav } from "@/components/sidebar-nav"
import { Logo } from "@/components/logo"
import { Button } from "@/components/ui/button"
import { Menu, X } from "lucide-react"
import { DepartmentInfo } from "@/components/department-info"
import { FaqSection } from "@/components/faq-section"
import { GpaCalculator } from "@/components/gpa-calculator"
import { ProgramInfo } from "@/components/program-info"
import { FacultyList } from "@/components/faculty-list"

interface ChatLayoutProps {
  children: React.ReactNode
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
}

export function ChatLayout({ children, sidebarOpen, setSidebarOpen }: ChatLayoutProps) {
  const [activeTab, setActiveTab] = useState<"info" | "faq" | "gpa" | "program" | "faculty">("info")

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <SidebarProvider open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <Sidebar className="border-r border-border bg-sidebar-background">
          <SidebarHeader className="flex items-center justify-between p-4">
            <Logo />
            <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setSidebarOpen(false)}>
              <X className="h-5 w-5" />
            </Button>
          </SidebarHeader>
          <SidebarContent>
            <SidebarNav activeTab={activeTab} setActiveTab={setActiveTab} />
            <div className="p-4">
              {activeTab === "info" && <DepartmentInfo />}
              {activeTab === "program" && <ProgramInfo />}
              {activeTab === "faculty" && <FacultyList />}
              {activeTab === "faq" && <FaqSection />}
              {activeTab === "gpa" && <GpaCalculator />}
            </div>
          </SidebarContent>
          <SidebarFooter className="p-4 text-xs text-muted-foreground">
            <div className="terminal-text">
              <p className="mb-1">SiSChatBot v1.0.0</p>
              <p>© {new Date().getFullYear()} Addis Ababa University - School of Information Science</p>
            </div>
          </SidebarFooter>
        </Sidebar>
        <div className="flex flex-1 flex-col overflow-hidden">
          <header className="flex h-14 items-center border-b border-border px-4">
            <Button variant="ghost" size="icon" className="mr-2" onClick={() => setSidebarOpen(true)}>
              <Menu className="h-5 w-5" />
            </Button>
            <h1 className="terminal-text text-lg font-bold retro-glow">SiSChatBot</h1>
            <div className="ml-auto text-xs text-muted-foreground terminal-text">
              <span className="inline-block h-2 w-2 rounded-full bg-green-500 mr-1"></span>
              ONLINE
            </div>
          </header>
          <div className="flex-1 overflow-hidden">{children}</div>
        </div>
      </SidebarProvider>
    </div>
  )
}
