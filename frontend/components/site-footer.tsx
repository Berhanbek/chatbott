import Link from "next/link"
import { Logo } from "@/components/logo"
import { BellIcon as BrandTelegram } from "lucide-react"

export function SiteFooter() {
  return (
    <footer className="border-t bg-muted/40">
      <div className="container py-6 px-4 md:px-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="flex items-center">
            <Logo />
          </div>

          <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm text-muted-foreground">
            <Link href="/department-info" className="hover:text-primary">
              Department Info
            </Link>
            <Link href="/program-details" className="hover:text-primary">
              Program Details
            </Link>
            <Link href="/faculty" className="hover:text-primary">
              Faculty & Staff
            </Link>
            <Link href="/gpa-calculator" className="hover:text-primary">
              GPA Calculator
            </Link>
            <a
              href="https://t.me/SISResourcesBot"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary flex items-center"
            >
              <BrandTelegram className="h-3 w-3 mr-1" />
              SIS Resource Bot
            </a>
          </div>
        </div>

        <div className="border-t border-border mt-4 pt-4 text-center text-xs text-muted-foreground">
          <p>
            © {new Date().getFullYear()} School of Information Science, Addis Ababa University. All rights reserved.
          </p>
          <p className="mt-1">
            Developed by <span className="text-primary font-medium">IS Seer</span>
          </p>
        </div>
      </div>
    </footer>
  )
}
