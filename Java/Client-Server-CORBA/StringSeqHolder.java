
/**
* StringSeqHolder.java .
* Generated by the IDL-to-Java compiler (portable), version "3.2"
* from ServerInterface.idl
* Friday, January 3, 2025 3:40:24 PM CET
*/

public final class StringSeqHolder implements org.omg.CORBA.portable.Streamable
{
  public String value[] = null;

  public StringSeqHolder ()
  {
  }

  public StringSeqHolder (String[] initialValue)
  {
    value = initialValue;
  }

  public void _read (org.omg.CORBA.portable.InputStream i)
  {
    value = StringSeqHelper.read (i);
  }

  public void _write (org.omg.CORBA.portable.OutputStream o)
  {
    StringSeqHelper.write (o, value);
  }

  public org.omg.CORBA.TypeCode _type ()
  {
    return StringSeqHelper.type ();
  }

}
