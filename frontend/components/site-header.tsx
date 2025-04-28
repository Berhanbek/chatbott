"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Logo } from "@/components/logo"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { Menu, MessageSquare, Users, Calculator, Info, BookOpen, Sun, Moon } from "lucide-react"
import { useState, useEffect } from "react"
import { useTheme } from "next-themes"

const navigation = [
  { name: "ChatBot", href: "/", icon: <MessageSquare className="h-4 w-4 mr-2" /> },
  { name: "Department Info", href: "/department-info", icon: <Info className="h-4 w-4 mr-2" /> },
  { name: "Program Details", href: "/program-details", icon: <BookOpen className="h-4 w-4 mr-2" /> },
  { name: "Faculty & Staff", href: "/faculty", icon: <Users className="h-4 w-4 mr-2" /> },
  { name: "GPA Calculator", href: "/gpa-calculator", icon: <Calculator className="h-4 w-4 mr-2" /> },
]

export function SiteHeader() {
  const pathname = usePathname()
  const [open, setOpen] = useState(false)
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-colors duration-300">
      <div className="container flex h-16 items-center justify-between">
        <Logo />

        <nav className="hidden md:flex items-center gap-4">
          {navigation.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={`text-sm font-medium transition-colors hover:text-primary flex items-center px-3 py-1.5 rounded-md ${
                pathname === item.href ? "text-primary bg-primary/10" : "text-muted-foreground hover:bg-muted/50"
              }`}
            >
              {item.icon}
              {item.name}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          {mounted && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="rounded-full"
            >
              {theme === "dark" ? (
                <Sun className="h-5 w-5 text-yellow-400" />
              ) : (
                <Moon className="h-5 w-5 text-slate-700" />
              )}
              <span className="sr-only">Toggle theme</span>
            </Button>
          )}

          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild className="md:hidden">
              <Button variant="ghost" size="icon">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[300px] sm:w-[400px]">
              <nav className="flex flex-col gap-4 mt-8">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    href={item.href}
                    onClick={() => setOpen(false)}
                    className={`text-lg font-medium transition-colors hover:text-primary flex items-center p-2 rounded-md ${
                      pathname === item.href ? "text-primary bg-primary/10" : "text-muted-foreground hover:bg-muted/50"
                    }`}
                  >
                    {item.icon}
                    {item.name}
                  </Link>
                ))}
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  )
}
