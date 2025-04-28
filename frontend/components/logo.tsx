import Image from "next/image"
import Link from "next/link"

export function Logo() {
  return (
    <Link href="/" className="flex items-center space-x-2">
      <Image src="/images/sis-logo.png" alt="SiS Logo" width={40} height={40} className="h-10 w-10" />
      <div className="flex flex-col">
        <span className="font-bold text-lg">SiS Super Bot</span>
        <span className="text-xs text-muted-foreground">School of Information Science</span>
      </div>
    </Link>
  )
}
